import functools
import math
import random
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use
import numpy as np
import torch
import torch.nn as nn
import pathos.multiprocessing as mp
import torch.multiprocessing as tmp
from timeit import default_timer as timer
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torchviz import make_dot
import os
from memory_profiler import profile

print(f"Using backend {plt.get_backend()}")

GRID_X = 40
GRID_Y = 40
GRID_Z = 40

INHOMOGENEOUS_ZERO_VECTOR = torch.tensor([0., 0., 0.])
REGULARISATION_FRACTION = 0.01
REGULARISATION_LAMBDA = 0.001
LEARNING_RATE = 0.001

MASTER_RAY_SAMPLE_POSITIONS_STRUCTURE = []
MASTER_VOXELS_STRUCTURE = []
VOXELS_NOT_USED = 0


class Camera:
    def __init__(self, focal_length, center, look_at):
        self.center = center
        self.basis = basis_from_depth(look_at, center)
        self.focal_length = focal_length
        camera_center = center.detach().clone()
        transposed_basis = torch.transpose(self.basis, 0, 1)
        camera_center[:3] = camera_center[
                            :3] * -1  # We don't want to multiply the homogenous coordinate component; it needs to remain 1
        camera_origin_translation = torch.eye(4, 4)
        camera_origin_translation[:, 3] = camera_center
        extrinsic_camera_parameters = torch.matmul(torch.inverse(transposed_basis), camera_origin_translation)
        intrinsic_camera_parameters = torch.tensor([[focal_length, 0., 0., 0.],
                                                    [0., focal_length, 0., 0.],
                                                    [0., 0., 1., 0.]])
        self.transform = torch.matmul(intrinsic_camera_parameters, extrinsic_camera_parameters)

    def to_2D(self, point):
        rendered_point = torch.matmul(self.transform, torch.transpose(point, 0, 1))
        point_z = rendered_point[2, 0]
        return rendered_point / point_z

    def viewing_angle(self):
        camera_basis_z = self.basis[2][:3]
        camera_basis_theta = math.atan(camera_basis_z[1] / camera_basis_z[0]) if (
                camera_basis_z[0] != 0) else math.pi / 2
        camera_basis_phi = math.atan((camera_basis_z[0] ** 2 + camera_basis_z[1] ** 2) / camera_basis_z[2]) if (
                camera_basis_z[2] != 0) else math.pi / 2
        return torch.tensor([camera_basis_theta, camera_basis_phi])


def camera_basis_from(camera_depth_z_vector):
    depth_vector = camera_depth_z_vector[:3]  # We just want the inhomogenous parts of the coordinates

    # This calculates the projection of the world z-axis onto the surface defined by the camera direction,
    # since we want to derive the coordinate system of the camera to be orthogonal without having
    # to calculate it manually.
    cartesian_z_vector = torch.tensor([0., 0., 1.])
    cartesian_z_projection_lambda = torch.dot(depth_vector, cartesian_z_vector) / torch.dot(
        depth_vector, depth_vector)
    camera_up_vector = cartesian_z_vector - cartesian_z_projection_lambda * depth_vector

    # This special case is for when the camera is directly pointing up or down, then
    # there is no way to decide which way to orient its up vector in the X-Y plane.
    # We choose to align the up veector with the X-axis in this case.
    if (torch.equal(camera_up_vector, INHOMOGENEOUS_ZERO_VECTOR)):
        camera_up_vector = torch.tensor([1., 0., 0.])
    print(f"Up vector is: {camera_up_vector}")
    # The camera coordinate system now has the direction of camera and the up direction of the camera.
    # We need to find the third vector which needs to be orthogonal to both the previous vectors.
    # Taking the cross product of these vectors gives us this third component
    camera_x_vector = torch.linalg.cross(depth_vector, camera_up_vector)
    inhomogeneous_basis = torch.stack([camera_x_vector, camera_up_vector, depth_vector, torch.tensor([0., 0., 0.])])
    homogeneous_basis = torch.hstack((inhomogeneous_basis, torch.tensor([[0.], [0.], [0.], [1.]])))
    homogeneous_basis[0] = unit_vector(homogeneous_basis[0])
    homogeneous_basis[1] = unit_vector(homogeneous_basis[1])
    homogeneous_basis[2] = unit_vector(homogeneous_basis[2])
    return homogeneous_basis


def basis_from_depth(look_at, camera_center):
    print(f"Looking at: {look_at}")
    print(f"Looking from: {camera_center}")
    depth_vector = torch.sub(look_at, camera_center)
    depth_vector[3] = 1.
    return camera_basis_from(depth_vector)


def unit_vector(camera_basis_vector):
    return camera_basis_vector / math.sqrt(
        pow(camera_basis_vector[0], 2) +
        pow(camera_basis_vector[1], 2) +
        pow(camera_basis_vector[2], 2))


def plot(style="bo"):
    return lambda p: plt.plot(p[0][0], p[1][0], style)


def generate_camera_angles(radius, look_at):
    camera_positions = []
    for phi in np.linspace(0, math.pi, 4):
        for theta in np.linspace(0, 2 * math.pi, 5):
            phi += math.pi / 4
            theta += math.pi / 4
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            x = 0 if abs(x) < 0.0001 else x
            y = 0 if abs(y) < 0.0001 else y
            z = 0 if abs(z) < 0.0001 else z
            camera_positions.append(torch.tensor([x, y, z, 0]))

    positions = (torch.stack(camera_positions).unique(dim=0)) + look_at
    print(positions)
    return positions


HALF_SQRT_3_BY_PI = 0.5 * math.sqrt(3. / math.pi)
HALF_SQRT_15_BY_PI = 0.5 * math.sqrt(15. / math.pi)
QUARTER_SQRT_15_BY_PI = 0.25 * math.sqrt(15. / math.pi)
QUARTER_SQRT_5_BY_PI = 0.25 * math.sqrt(5. / math.pi)

Y_0_0 = 0.5 * math.sqrt(1. / math.pi)
Y_m1_1 = lambda theta, phi: HALF_SQRT_3_BY_PI * math.sin(theta) * math.sin(phi)
Y_0_1 = lambda theta, phi: HALF_SQRT_3_BY_PI * math.cos(theta)
Y_1_1 = lambda theta, phi: HALF_SQRT_3_BY_PI * math.sin(theta) * math.cos(phi)
Y_m2_2 = lambda theta, phi: HALF_SQRT_15_BY_PI * math.sin(theta) * math.cos(phi) * math.sin(
    theta) * math.sin(phi)
Y_m1_2 = lambda theta, phi: HALF_SQRT_15_BY_PI * math.sin(theta) * math.sin(phi) * math.cos(theta)
Y_0_2 = lambda theta, phi: QUARTER_SQRT_5_BY_PI * (3 * math.cos(theta) * math.cos(theta) - 1)
Y_1_2 = lambda theta, phi: HALF_SQRT_15_BY_PI * math.sin(theta) * math.cos(phi) * math.cos(theta)
Y_2_2 = lambda theta, phi: QUARTER_SQRT_15_BY_PI * (
        pow(math.sin(theta) * math.cos(phi), 2) - pow(math.sin(theta) * math.sin(phi), 2))


def harmonic(C_0_0, C_m1_1, C_0_1, C_1_1, C_m2_2, C_m1_2, C_0_2, C_1_2, C_2_2):
    return lambda theta, phi: C_0_0 * Y_0_0 + C_m1_1 * Y_m1_1(theta, phi) + C_0_1 * Y_0_1(theta, phi) + C_1_1 * Y_1_1(
        theta, phi) + C_m2_2 * Y_m2_2(theta, phi) + C_m1_2 * Y_m1_2(theta, phi) + C_0_2 * Y_0_2(theta,
                                                                                                phi) + C_1_2 * Y_1_2(
        theta, phi) + C_2_2 * Y_2_2(theta, phi)


def rgb_harmonics(rgb_harmonic_coefficients):
    red_harmonic = harmonic(*rgb_harmonic_coefficients[:9])
    green_harmonic = harmonic(*rgb_harmonic_coefficients[9:18])
    blue_harmonic = harmonic(*rgb_harmonic_coefficients[18:])
    return (red_harmonic, green_harmonic, blue_harmonic)


class Voxel:
    NUM_INTERPOLATING_NEIGHBOURS = 8
    DEFAULT_OPACITY = 0.05

    @staticmethod
    def default_voxel(requires_grad=True):
        # voxel = torch.cat([torch.tensor([0.25]), torch.ones(VoxelGrid.VOXEL_DIMENSION - 1)])
        return lambda: torch.tensor(Voxel.uniform_harmonic(), requires_grad=requires_grad)

    @staticmethod
    def random_coloured_voxel(requires_grad=True):
        return lambda: torch.tensor(np.concatenate(([0.0002], np.random.rand(VoxelGrid.VOXEL_DIMENSION - 1))),
                                    requires_grad=requires_grad)

    @staticmethod
    def uniform_harmonic():
        return [1.] + ([1.] + [0.] * (VoxelGrid.PER_CHANNEL_DIMENSION - 1)) * 3

    @staticmethod
    def random_harmonic_coefficient_set():
        return ([random.random()] + [0.] * (VoxelGrid.PER_CHANNEL_DIMENSION - 1))

    @staticmethod
    def uniform_harmonic_random_colour(requires_grad=True):
        return lambda: torch.tensor([
                                        0.1] + Voxel.random_harmonic_coefficient_set() + Voxel.random_harmonic_coefficient_set() + Voxel.random_harmonic_coefficient_set(),
                                    requires_grad=requires_grad)

    @staticmethod
    def occupied_voxel(requires_grad=True):
        return lambda: torch.tensor(Voxel.uniform_harmonic(), requires_grad=requires_grad)

    @staticmethod
    def empty_voxel(requires_grad=True):
        return lambda: torch.zeros([VoxelGrid.VOXEL_DIMENSION], requires_grad=requires_grad)

    @staticmethod
    def like_voxel(prototype_voxel):
        return lambda: prototype_voxel.clone()


class Ray:
    def __init__(self, num_samples, view_point, ray_sample_positions, voxel_positions, voxels):
        self.num_samples = num_samples
        self.ray_sample_positions = ray_sample_positions
        self.view_point = view_point
        self.voxels = voxels
        self.voxel_positions = voxel_positions
        if num_samples != len(ray_sample_positions):
            print(f"WARNING: num_samples = {num_samples}, sample_positions = {ray_sample_positions}")

    def at(self, index):
        start = index * Voxel.NUM_INTERPOLATING_NEIGHBOURS
        end = start + Voxel.NUM_INTERPOLATING_NEIGHBOURS
        return self.ray_sample_positions[index], \
               self.voxel_positions[start: end], \
               self.voxels[start: end]


class VoxelAccess:
    def __init__(self, view_points, ray_sample_positions, voxel_pointers, all_voxels, all_voxel_positions):
        self.ray_sample_positions = ray_sample_positions
        self.view_points = view_points
        self.voxel_positions = all_voxel_positions
        self.all_voxels = all_voxels
        self.voxel_pointers = voxel_pointers

    def for_ray(self, ray_index):
        ptr = self.voxel_pointers[ray_index]
        start, end, num_samples = ptr
        return Ray(num_samples, self.view_points[ray_index],
                   self.ray_sample_positions[int(start / 8): int(end / 8)],
                   self.voxel_positions[start:end],
                   self.all_voxels[start:end])


class VoxelGrid:
    VOXEL_DIMENSION = 28
    PER_CHANNEL_DIMENSION = 9

    def __init__(self, world_tensor):
        self.grid_x, self.grid_y, self.grid_z = world_tensor.shape
        self.voxel_grid = world_tensor

    def world_x(self):
        return self.grid_x

    def world_y(self):
        return self.grid_y

    def world_z(self):
        return self.grid_z

    @staticmethod
    def build_empty_world(x, y, z):
        return VoxelGrid.new(x, y, z, Voxel.empty_voxel())

    @staticmethod
    def build_random_world(x, y, z):
        return VoxelGrid.new(x, y, z, Voxel.random_coloured_voxel())

    @staticmethod
    def build_with_voxel(x, y, z, prototype_voxel):
        return VoxelGrid.new(x, y, z, Voxel.like_voxel(prototype_voxel))

    @staticmethod
    def copy_from(world_tensor):
        x, y, z = world_tensor.shape
        new_world = VoxelGrid.build_empty_world(x, y, z)
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    new_world.set((i, j, k, 1), world_tensor[i, j, k])
        return new_world

    @classmethod
    def from_tensor(cls, world_tensor):
        return cls(world_tensor)

    @classmethod
    def as_parameter(cls, world_tensor, model):
        x, y, z = world_tensor.shape
        new_world = VoxelGrid.build_empty_world(x, y, z)
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    parameter = nn.Parameter(world_tensor[i, j, k])
                    new_world.set((i, j, k, 1), parameter)
                    model.register_parameter(f"{(i, j, k)}", parameter)
        return new_world

    @classmethod
    def new(cls, x, y, z, make_voxel):
        voxel_grid = np.ndarray((x, y, z), dtype=list)
        for i in range(x):
            for j in range(y):
                for k in range(y):
                    voxel_grid[i, j, k] = make_voxel()
        return cls(voxel_grid)

    def at(self, x, y, z):
        if self.is_outside(x, y, z):
            return Voxel.empty_voxel(requires_grad=False)()
        else:
            return self.voxel_grid[int(x), int(y), int(z)]

    def set(self, position, voxel):
        x, y, z, _ = position
        if self.is_outside(x, y, z):
            return
        else:
            self.voxel_grid[int(x), int(y), int(z)] = voxel

    def is_inside(self, x, y, z):
        if (0 <= x < self.grid_x and
                0 <= y < self.grid_y and
                0 <= z < self.grid_z):
            return True

    def is_outside(self, x, y, z):
        return not self.is_inside(x, y, z)

    def channel_opacity(self, position_distance_density_color_tensors, viewing_angle):
        number_of_samples = len(position_distance_density_color_tensors)
        transmittances = list(map(lambda i: functools.reduce(
            lambda acc, j: acc + math.exp(
                - position_distance_density_color_tensors[j, 4] * position_distance_density_color_tensors[j, 3]),
            range(0, i), 0.), range(1, number_of_samples + 1)))
        color_densities = torch.zeros([3])
        for index, transmittance in enumerate(transmittances):
            if (position_distance_density_color_tensors[index, 4] == 0.):
                continue
            # density += (transmittance - transmittances[index + 1]) * position_distance_density_color_vectors[index, 5]
            red_harmonic, green_harmonic, blue_harmonic = rgb_harmonics(
                position_distance_density_color_tensors[index, 5:])
            r = red_harmonic(viewing_angle[0], viewing_angle[1])
            g = green_harmonic(viewing_angle[0], viewing_angle[1])
            b = blue_harmonic(viewing_angle[0], viewing_angle[1])
            base_transmittance = transmittance * (1. - torch.exp(
                - position_distance_density_color_tensors[index, 4] * position_distance_density_color_tensors[
                    index, 3]))

            # Stacking instead of cat-ing to preserve gradient
            color_densities += torch.stack([base_transmittance * r, base_transmittance * g, base_transmittance * b])

        return color_densities

    def build_solid_cube(self, cube_spec):
        x, y, z, dx, dy, dz = cube_spec
        for i in range(x, x + dx):
            for j in range(y, y + dy):
                for k in range(z, z + dz):
                    self.voxel_grid[i, j, k] = Voxel.occupied_voxel()()

    def build_monochrome_hollow_cube(self, cube_spec):
        self.build_hollow_cube2(Voxel.default_voxel(), cube_spec)

    def build_hollow_cube(self, make_voxel):
        self.build_hollow_cube2(make_voxel, torch.tensor([10, 10, 10, 20, 20, 20]))

    def build_hollow_cube2(self, make_voxel, cube_spec):
        x, y, z, dx, dy, dz = cube_spec
        voxel_1, voxel_2, voxel_3, voxel_4, voxel_5, voxel_6 = make_voxel(), make_voxel(), make_voxel(), make_voxel(), make_voxel(), make_voxel()
        for i in range(x, x + dx + 1):
            for j in range(y, y + dy + 1):
                self.voxel_grid[i, j, z] = make_voxel()
        for i in range(x, x + dx + 1):
            for j in range(y, y + dy + 1):
                self.voxel_grid[i, j, z + dz] = make_voxel()
        for i in range(y, y + dy + 1):
            for j in range(z, z + dz + 1):
                self.voxel_grid[x, i, j] = make_voxel()
        for i in range(y, y + dy + 1):
            for j in range(z, z + dz + 1):
                self.voxel_grid[x + dx, i, j] = make_voxel()
        for i in range(z, z + dz + 1):
            for j in range(x, x + dx + 1):
                self.voxel_grid[j, y, i] = make_voxel()
        for i in range(z, z + dz + 1):
            for j in range(x, x + dx + 1):
                self.voxel_grid[j, y + dy, i] = make_voxel()

    def build_hollow_cube_with_randomly_coloured_sides(self, make_voxel, cube_spec):
        x, y, z, dx, dy, dz = cube_spec
        voxel_1, voxel_2, voxel_3, voxel_4, voxel_5, voxel_6 = make_voxel(), make_voxel(), make_voxel(), make_voxel(), make_voxel(), make_voxel()
        for i in range(x, x + dx + 1):
            for j in range(y, y + dy + 1):
                self.voxel_grid[i, j, z] = voxel_1
        for i in range(x, x + dx + 1):
            for j in range(y, y + dy + 1):
                self.voxel_grid[i, j, z + dz] = voxel_2
        for i in range(y, y + dy + 1):
            for j in range(z, z + dz + 1):
                self.voxel_grid[x, i, j] = voxel_3
        for i in range(y, y + dy + 1):
            for j in range(z, z + dz + 1):
                self.voxel_grid[x + dx, i, j] = voxel_4
        for i in range(z, z + dz + 1):
            for j in range(x, x + dx + 1):
                self.voxel_grid[j, y, i] = voxel_5
        for i in range(z, z + dz + 1):
            for j in range(x, x + dx + 1):
                self.voxel_grid[j, y + dy, i] = voxel_6

    def density(self, ray_samples_with_distances, viewing_angle):
        global MASTER_VOXELS_STRUCTURE
        collected_intensities = []
        for ray_sample in ray_samples_with_distances:
            x = ray_sample[0]
            y = ray_sample[1]
            z = ray_sample[2]
            x_0, x_1 = int(x), int(x) + 1
            y_0, y_1 = int(y), int(y) + 1
            z_0, z_1 = int(z), int(z) + 1
            x_d = (x - x_0) / (x_1 - x_0)
            y_d = (y - y_0) / (y_1 - y_0)
            z_d = (z - z_0) / (z_1 - z_0)
            c_000 = self.at(x_0, y_0, z_0)
            c_001 = self.at(x_0, y_0, z_1)
            c_010 = self.at(x_0, y_1, z_0)
            c_011 = self.at(x_0, y_1, z_1)
            c_100 = self.at(x_1, y_0, z_0)
            c_101 = self.at(x_1, y_0, z_1)
            c_110 = self.at(x_1, y_1, z_0)
            c_111 = self.at(x_1, y_1, z_1)
            c_00 = c_000 * (1 - x_d) + c_100 * x_d
            c_01 = c_001 * (1 - x_d) + c_101 * x_d
            c_10 = c_010 * (1 - x_d) + c_110 * x_d
            c_11 = c_011 * (1 - x_d) + c_111 * x_d

            c_0 = c_00 * (1 - y_d) + c_10 * y_d
            c_1 = c_01 * (1 - y_d) + c_11 * y_d

            c = c_0 * (1 - z_d) + c_1 * z_d
            MASTER_VOXELS_STRUCTURE += [c_000, c_001, c_010, c_011, c_100, c_101, c_110, c_111]

            collected_intensities.append(c)
        return self.channel_opacity(torch.cat([ray_samples_with_distances, torch.stack(collected_intensities)], 1),
                                    viewing_angle)


def neighbours(x, y, z, world):
    x0 = int(x)
    y0 = int(y)
    z0 = int(z)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    c000 = world.at(x0, y0, z0)
    c001 = world.at(x0, y0, z1)
    c010 = world.at(x0, y1, z0)
    c011 = world.at(x0, y1, z1)
    c100 = world.at(x1, y0, z0)
    c101 = world.at(x1, y0, z1)
    c110 = world.at(x1, y1, z0)
    c111 = world.at(x1, y1, z1)

    return ([c000, c001, c010, c011, c100, c101, c110, c111], torch.tensor([[x0, y0, z0],
                                                                            [x0, y0, z1],
                                                                            [x0, y1, z0],
                                                                            [x0, y1, z1],
                                                                            [x1, y0, z0],
                                                                            [x1, y0, z1],
                                                                            [x1, y1, z0],
                                                                            [x1, y1, z1]]))


def density_split(ray_sample_distances, ray, viewing_angle, world):
    collected_intensities = []
    for index, distance in enumerate(ray_sample_distances):
        ray_sample_position, voxel_positions, voxels = ray.at(index)
        if len(voxels) == 0:
            return torch.tensor([0., 0., 0.])
        c_000, c_001, c_010, c_011, c_100, c_101, c_110, c_111 = voxels
        x, y, z = ray_sample_position
        x_0, x_1 = int(x), int(x) + 1
        y_0, y_1 = int(y), int(y) + 1
        z_0, z_1 = int(z), int(z) + 1
        x_d = (x - x_0) / (x_1 - x_0)
        y_d = (y - y_0) / (y_1 - y_0)
        z_d = (z - z_0) / (z_1 - z_0)

        c_00 = c_000 * (1 - x_d) + c_100 * x_d
        c_01 = c_001 * (1 - x_d) + c_101 * x_d
        c_10 = c_010 * (1 - x_d) + c_110 * x_d
        c_11 = c_011 * (1 - x_d) + c_111 * x_d

        c_0 = c_00 * (1 - y_d) + c_10 * y_d
        c_1 = c_01 * (1 - y_d) + c_11 * y_d

        c = c_0 * (1 - z_d) + c_1 * z_d

        collected_intensities.append(c)

    return channel_opacity_split(torch.cat([ray_sample_distances, torch.stack(collected_intensities)], 1),
                                 viewing_angle)


def channel_opacity_split(distance_density_color_tensors, viewing_angle):
    number_of_samples = len(distance_density_color_tensors)
    transmittances = list(map(lambda i: functools.reduce(
        lambda acc, j: acc + math.exp(
            - distance_density_color_tensors[j, 1] * distance_density_color_tensors[j, 0]),
        range(0, i), 0.), range(1, number_of_samples + 1)))
    color_densities = torch.zeros([3])
    for index, transmittance in enumerate(transmittances):
        if (distance_density_color_tensors[index, 1] == 0.):
            continue

        red_harmonic, green_harmonic, blue_harmonic = rgb_harmonics(
            distance_density_color_tensors[index, 2:])
        r = red_harmonic(viewing_angle[0], viewing_angle[1])
        g = green_harmonic(viewing_angle[0], viewing_angle[1])
        b = blue_harmonic(viewing_angle[0], viewing_angle[1])
        base_transmittance = transmittance * (1. - torch.exp(
            - distance_density_color_tensors[index, 1] * distance_density_color_tensors[
                index, 0]))

        # Stacking instead of cat-ing to preserve gradient
        color_densities += torch.stack([base_transmittance * r, base_transmittance * g, base_transmittance * b])

    return color_densities


class ClampingFunctions:
    SIGMOID = nn.Sigmoid()
    CLAMP = lambda t: torch.clamp(t, min=0, max=1)


class Renderer:
    def __init__(self, world, camera, view_spec, ray_spec):
        self.world = world
        self.camera = camera
        self.ray_length = ray_spec[0]
        self.num_ray_samples = ray_spec[1]
        self.x_1, self.x_2 = view_spec[0], view_spec[1]
        self.y_1, self.y_2 = view_spec[2], view_spec[3]
        self.num_view_samples_x = view_spec[4]
        self.num_view_samples_y = view_spec[5]

    def dummy(self):
        return 1

    def render_from_ray_from_angle(self, viewing_angle):
        return lambda ray: self.render_from_ray(ray, viewing_angle)

    def render_from_ray(self, ray, viewing_angle, clamping_function=ClampingFunctions.SIGMOID):
        # print(f"Wall clock in render_from_ray() is {timer()}")
        ray_sample_positions = ray.ray_sample_positions
        unique_ray_samples = ray_sample_positions
        view_x, view_y = ray.view_point

        if (len(unique_ray_samples) <= 1):
            return torch.tensor([view_x, view_y, 0., 0., 0.])

        t1 = unique_ray_samples[:-1]
        t2 = unique_ray_samples[1:]
        consecutive_sample_distances = (t1 - t2).pow(2).sum(1).sqrt()

        # Make 1D tensor into 2D tensor
        # List of tensors, each entry is distance from i-th sample to the next sample
        ray_sample_distances = torch.reshape(consecutive_sample_distances, (-1, 1))
        color_densities = density_split(ray_sample_distances, ray, viewing_angle, self.world)
        color_tensor = clamping_function(color_densities)

        if (view_x < self.x_1 or view_x > self.x_2
                or view_y < self.y_1 or view_y > self.y_2):
            print(f"Warning: bad generation: {view_x}, {view_y}")

        # print(color_tensor)
        return torch.cat([torch.tensor([view_x, view_y]), color_tensor])

    def render_from_rays(self, voxel_access):
        X, Y = 0, 1
        RED_CHANNEL, GREEN_CHANNEL, BLUE_CHANNEL = 2, 3, 4
        camera = self.camera
        # composite_colour_tensors = self.render_parallel(voxel_access, camera)
        composite_colour_tensors = self.render_serial(voxel_access, camera)
        red_channel = composite_colour_tensors[:, [X, Y, RED_CHANNEL]]
        green_channel = composite_colour_tensors[:, [X, Y, GREEN_CHANNEL]]
        blue_channel = composite_colour_tensors[:, [X, Y, BLUE_CHANNEL]]
        print("Done volumetric calculations from rays!!")
        return (red_channel, green_channel, blue_channel)

    def render_serial(self, voxel_access, camera):
        viewing_angle = camera.viewing_angle()
        num_view_points = len(voxel_access.view_points)
        composite_colour_tensors = torch.stack(list(
            map(lambda index: self.render_from_ray(voxel_access.for_ray(index), viewing_angle),
                range(num_view_points))))
        return composite_colour_tensors

    def render_from_angle(self, ray):
        return self.render_from_ray(ray, self.camera.viewing_angle())

    def render_parallel(self, voxel_access, camera):
        viewing_angle = camera.viewing_angle()
        num_view_points = len(voxel_access.view_points)
        workers = os.cpu_count()
        p = tmp.Pool(workers)
        start_copy_rays = timer()
        rays = list(map(lambda i: voxel_access.for_ray(i), range(num_view_points)))
        end_copy_rays = timer()
        print(f"Copying rays took {end_copy_rays - start_copy_rays}")
        print(f"Wall clock is {timer()}")
        start_render_rays = timer()
        # responses = p.map(lambda ray: self.render_from_ray(ray, viewing_angle), rays)
        responses = p.map(self.render_from_angle, rays)
        p.close()
        p.join()
        end_render_rays = timer()
        print(f"Actual rendering took {end_render_rays - start_render_rays}")
        composite_colour_tensors = torch.stack(list(responses))
        return composite_colour_tensors

    @staticmethod
    def initialise_plt(plt):
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0
        figure = plt.figure(f"{random.random()}", frameon=False)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.axis("equal")
        plt.style.use("dark_background")
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.axis("off")
        return figure

    def build_rays(self, ray_intersection_weights):
        camera = self.camera
        camera_basis_x = camera.basis[0][:3]
        camera_basis_y = camera.basis[1][:3]
        camera_basis_z = camera.basis[2][:3]
        camera_center_inhomogenous = camera.center[:3]
        all_voxel_positions = []
        view_points = []
        voxel_pointers = []
        all_voxels = []
        ray_sample_positions = []
        view_screen_origin = camera_basis_z * camera.focal_length + camera_center_inhomogenous
        counter = 0
        for ray_intersection_weight in ray_intersection_weights:
            ray_screen_intersection = camera_basis_x * ray_intersection_weight[0] + \
                                      camera_basis_y * ray_intersection_weight[1] + view_screen_origin
            unit_ray = unit_vector(ray_screen_intersection - camera_center_inhomogenous)
            view_x, view_y = ray_intersection_weight[0], ray_intersection_weight[1]
            num_intersecting_voxels = 0

            all_voxels_per_ray = []
            all_voxel_positions_per_ray = []
            ray_sample_positions_per_ray = []
            for k in np.linspace(0, self.ray_length, self.num_ray_samples):
                ray_endpoint = camera_center_inhomogenous + unit_ray * k
                ray_x, ray_y, ray_z = ray_endpoint
                if (self.world.is_outside(ray_x, ray_y, ray_z)):
                    continue
                # We are in the box
                interpolating_voxels, interpolating_voxel_positions = neighbours(ray_x, ray_y, ray_z, self.world)
                num_intersecting_voxels += 1
                all_voxels_per_ray += interpolating_voxels
                all_voxel_positions_per_ray += interpolating_voxel_positions
                ray_sample_positions_per_ray.append(torch.tensor([ray_x, ray_y, ray_z]))
            if (num_intersecting_voxels <= 1):
                continue
            all_voxels += all_voxels_per_ray
            all_voxel_positions += all_voxel_positions_per_ray
            ray_sample_positions += ray_sample_positions_per_ray

            view_points.append((view_x, view_y))
            voxel_pointers.append((counter, counter + 8 * num_intersecting_voxels, num_intersecting_voxels))
            counter += 8 * num_intersecting_voxels

            if (view_x < self.x_1 or view_x > self.x_2
                    or view_y < self.y_1 or view_y > self.y_2):
                print(f"Warning: bad generation: {view_x}, {view_y}")
        print("Done building candidate rays!!")

        return VoxelAccess(view_points, torch.stack(ray_sample_positions), voxel_pointers, all_voxels,
                           all_voxel_positions)

    def render(self, plt, clamping_function=ClampingFunctions.SIGMOID, text=None):
        RED_CHANNEL, GREEN_CHANNEL, BLUE_CHANNEL = 0, 1, 2
        global VOXELS_NOT_USED
        global MASTER_RAY_SAMPLE_POSITIONS_STRUCTURE
        red_image = []
        green_image = []
        blue_image = []
        camera = self.camera
        camera_basis_x = camera.basis[0][:3]
        camera_basis_y = camera.basis[1][:3]
        camera_basis_z = camera.basis[2][:3]
        viewing_angle = camera.viewing_angle()
        camera_center_inhomogenous = camera.center[:3]

        Renderer.initialise_plt(plt)
        print(f"Camera basis={camera.basis}")
        view_screen_origin = camera_basis_z * camera.focal_length + camera_center_inhomogenous
        print(f"View screen origin={view_screen_origin}")
        for i in np.linspace(self.x_1, self.x_2, self.num_view_samples_x):
            red_column = []
            green_column = []
            blue_column = []
            for j in np.linspace(self.y_1, self.y_2, self.num_view_samples_y):
                ray_screen_intersection = camera_basis_x * i + camera_basis_y * j + view_screen_origin
                unit_ray = unit_vector(ray_screen_intersection - camera_center_inhomogenous)
                # print(f"Camera basis is {camera.basis}, Camera center is {camera_center_inhomogenous}, intersection is {ray_screen_intersection}, Unit ray is [{unit_ray}]")
                ray_samples = []
                # To remove artifacts, set ray step samples to be higher, like 200
                for k in np.linspace(0, self.ray_length, self.num_ray_samples):
                    ray_endpoint = camera_center_inhomogenous + unit_ray * k
                    ray_x, ray_y, ray_z = ray_endpoint
                    if (self.world.is_outside(ray_x, ray_y, ray_z)):
                        # print(
                        # f"Skipping [{ray_x},{ray_y},{ray_z}], k={k}, unit ray={unit_ray}, camera is {camera_center_inhomogenous}")
                        continue
                    # We are in the box
                    ray_samples.append([ray_x, ray_y, ray_z])
                    # print(
                    #     f"Sample at ({[ray_x, ray_y, ray_z]}), voxel value here is {self.world.at(ray_x, ray_y, ray_z)}")

                # unique_ray_samples = torch.unique(torch.tensor(ray_samples), dim=0)
                unique_ray_samples = torch.tensor(ray_samples)
                if (len(unique_ray_samples) <= 1):
                    red_column.append(torch.tensor(0))
                    green_column.append(torch.tensor(0))
                    blue_column.append(torch.tensor(0))
                    plt.plot(i, j, marker="o", color=[0, 0, 0])
                    # print("Too few")
                    continue

                MASTER_RAY_SAMPLE_POSITIONS_STRUCTURE += unique_ray_samples
                VOXELS_NOT_USED += 8
                t1 = unique_ray_samples[:-1]
                t2 = unique_ray_samples[1:]
                consecutive_sample_distances = (t1 - t2).pow(2).sum(1).sqrt()
                # print(consecutive_sample_distances)

                # Make 1D tensor into 2D tensor
                ray_samples_with_distances = torch.cat([t1, torch.reshape(consecutive_sample_distances, (-1, 1))], 1)
                # print(ray_samples_with_distances)
                color_densities = self.world.density(ray_samples_with_distances, viewing_angle)
                # print(color_densities)

                # color_tensor = torch.clamp(color_densities, min=0, max=1)
                color_tensor = clamping_function(color_densities)
                plt.plot(i, j, marker="o", color=color_tensor.detach().numpy())
                red_column.append(color_tensor[RED_CHANNEL])
                green_column.append(color_tensor[GREEN_CHANNEL])
                blue_column.append(color_tensor[BLUE_CHANNEL])
            red_image.append(torch.tensor(red_column))
            green_image.append(torch.tensor(green_column))
            blue_image.append(torch.tensor(blue_column))

        # Flip to prevent image being rendered upside down when saved to a file
        red_image_tensor = torch.flip(torch.stack(red_image).t(), [0])
        green_image_tensor = torch.flip(torch.stack(green_image).t(), [0])
        blue_image_tensor = torch.flip(torch.stack(blue_image).t(), [0])

        if (text is not None):
            plt.text(0.5, 0.5, text)
        plt.show()
        print("Done rendering in full!!")
        return (red_image_tensor, green_image_tensor, blue_image_tensor)

    def plot_from_image(self, image_data, plt, text=None):
        Renderer.initialise_plt(plt)
        red_render_channel, green_render_channel, blue_render_channel = image_data.detach().numpy()
        width, height = red_render_channel.shape
        for i in range(width):
            for j in range(height):
                plt.plot(i, height - 1 - j, marker="o",
                         color=[red_render_channel[j, i], green_render_channel[j, i], blue_render_channel[j, i]])
        if (text is not None):
            plt.text(0.5, 0.5, text, fontsize=14)
        plt.show()


def stochastic_samples(num_stochastic_samples, view_spec):
    x_1, x_2 = view_spec[0], view_spec[1]
    y_1, y_2 = view_spec[2], view_spec[3]
    view_length = x_2 - x_1
    view_height = y_2 - y_1

    # Need to convert the range [Random(0,1), Random(0,1)] into bounds of [[x1, x2], [y1, y2]]
    ray_intersection_weights = list(
        map(lambda x: torch.mul(torch.rand(2), torch.tensor([view_length, view_height])) + torch.tensor(
            [x_1, y_1]), list(range(0, num_stochastic_samples))))
    return ray_intersection_weights


def fullscreen_samples(view_spec):
    x_1, x_2 = view_spec[0], view_spec[1]
    y_1, y_2 = view_spec[2], view_spec[3]
    num_view_samples_x = view_spec[4]
    num_view_samples_y = view_spec[5]

    ray_intersection_weights = []
    for i in np.linspace(x_1, x_2, num_view_samples_x):
        for j in np.linspace(y_1, y_2, num_view_samples_y):
            ray_intersection_weights.append(torch.tensor([i, j]))
    return ray_intersection_weights


def camera_to_image(x, y, view_spec):
    view_x1, view_x2, view_y1, view_y2, num_rays_x, num_rays_y = view_spec
    step_x = (view_x2 - view_x1) / num_rays_x
    step_y = (view_y2 - view_y1) / num_rays_y

    # (view_y2 - y) implies we are flipping the Y-axis
    image_x = int((x - view_x1) / step_x)
    image_y = int((view_y2 - y) / step_y)

    # In the above calculation, [-1,1] maps to [0, num_rays]. Only +1 maps to num_rays.
    # We need to handle that isolated case and decrement by 1 to bring into the range
    #  of valid indices
    image_x = image_x if image_x < num_rays_x else image_x - 1
    image_y = image_y if image_y < num_rays_y else image_x - 1
    return (image_x, image_y)


def samples_to_image(red_samples, green_samples, blue_samples, view_spec):
    X, Y, INTENSITY = 0, 1, 2
    view_x1, view_x2, view_y1, view_y2, num_rays_x, num_rays_y = view_spec
    red_render_channel = torch.zeros([num_rays_y, num_rays_x])
    green_render_channel = torch.zeros([num_rays_y, num_rays_x])
    blue_render_channel = torch.zeros([num_rays_y, num_rays_x])
    for index, pixel in enumerate(red_samples):
        # print(
        #     f"({view_y2 - pixel[1]}, {pixel[0] - view_x1}) -> ({int((view_y2 - pixel[1]) / step_y)}, {int((pixel[0] - view_x1) / step_x)}), {pixel[2]}")
        x, y = camera_to_image(pixel[X], pixel[Y], view_spec)
        red_render_channel[y - 1, x - 1] = red_samples[index][INTENSITY]
        green_render_channel[y - 1, x - 1] = green_samples[index][INTENSITY]
        blue_render_channel[y - 1, x - 1] = blue_samples[index][INTENSITY]
    image_data = torch.stack([red_render_channel, green_render_channel, blue_render_channel])
    return image_data


def camera_to_image_test(x, y, view_spec):
    return int(x), int(y)


def mse(rendered_channel, true_channel, view_spec):
    # true_channel = torch.ones([2, 2]) * 10
    small_diffs = 0
    medium_diffs = 0
    large_diffs = 0
    channel_total_error = torch.tensor(0.)
    for point in rendered_channel:
        x, y, intensity = point
        intensity = intensity
        image_x, image_y = camera_to_image(x, y, view_spec)
        pixel_error = (true_channel[image_y, image_x] - intensity).pow(2)
        # print(pixel_error)
        if (pixel_error <= 0.001):
            small_diffs += 1
        elif (pixel_error > 0.001 and pixel_error <= 0.01):
            medium_diffs += 1
        else:
            large_diffs += 1
        channel_total_error += pixel_error

    # print(f"Small diffs = {small_diffs}")
    # print(f"Medium diffs = {medium_diffs}")
    # print(f"Large diffs = {large_diffs}")
    return channel_total_error / len(rendered_channel)


def tv_for_voxel(voxel_accessor, world):
    all_voxels = voxel_accessor.all_voxels
    voxel_positions = voxel_accessor.voxel_positions
    index = int(random.random() * len(all_voxels))
    position = voxel_positions[index]
    voxel = all_voxels[index]
    x_plus_1 = position + torch.tensor([1., 0., 0.])
    y_plus_1 = position + torch.tensor([0., 1., 0.])
    z_plus_1 = position + torch.tensor([0., 0., 1.])
    voxel_x1 = world.at(*x_plus_1)
    voxel_y1 = world.at(*y_plus_1)
    voxel_z1 = world.at(*z_plus_1)
    delta_x = ((voxel - voxel_x1) / (256 / world.world_x())).pow(2)
    delta_y = ((voxel - voxel_y1) / (256 / world.world_y())).pow(2)
    delta_z = ((voxel - voxel_z1) / (256 / world.world_z())).pow(2)

    sqrt__sum = (delta_x + delta_y + delta_z + 0.0001).sqrt().sum()
    if (math.isnan(sqrt__sum)):
        print("[WARNING] NaN in TV regularisation term")
        print(
            f"Sqrt sum={sqrt__sum}, Source voxel={voxel}, Positions are: {(x_plus_1, y_plus_1, z_plus_1)}, Voxels = {(voxel_x1, voxel_y1, voxel_z1)}, Deltas={(delta_x, delta_y, delta_z)}")
    return sqrt__sum


def tv_term(voxel_accessor, world):
    num_voxels_to_include = int(REGULARISATION_FRACTION * len(voxel_accessor.all_voxels))
    return torch.stack(
        list(map(lambda i: tv_for_voxel(voxel_accessor, world), list(range(num_voxels_to_include))))).mean()


NUM_STOCHASTIC_RAYS = 1500


def modify_grad(parameter_world, voxel_access):
    for i in range(parameter_world.world_x()):
        for j in range(parameter_world.world_y()):
            for k in range(parameter_world.world_z()):
                parameter_world.at(i, j, k).requires_grad = False

    for ray_index, view_point in enumerate(voxel_access.view_points):
        ray = voxel_access.for_ray(ray_index)
        for i in range(ray.num_samples):
            sample_position, voxel_positions, voxels = ray.at(i)
            for voxel_position in voxel_positions:
                x, y, z = voxel_position
                if (x < 0 or x > GRID_X - 1 or
                        y < 0 or y > GRID_Y - 1 or
                        z < 0 or z > GRID_Z - 1):
                    continue
                parameter_world.at(x, y, z).requires_grad = True


class PlenoxelModel(nn.Module):
    def __init__(self, world):
        super().__init__()

        # self.parameter_voxels = nn.Parameter(torch.tensor(world.voxel_grid), requires_grad=True)
        self.parameter_world = VoxelGrid.as_parameter(world.voxel_grid, self)
        # voxel_access = PlenoxelModel.run(self.parameter_world, [camera, view_spec, ray_spec])
        # self.voxels = nn.Parameter(torch.stack(self.voxel_access.all_voxels))
        # modify_grad(self.parameter_voxels, voxel_access, world)

    @staticmethod
    def run(world, input):
        camera, view_spec, ray_spec = input
        renderer = Renderer(world, camera, view_spec, ray_spec)
        # This draws stochastic rays and returns a set of samples with colours
        num_stochastic_rays = NUM_STOCHASTIC_RAYS
        voxel_access = renderer.build_rays(stochastic_samples(num_stochastic_rays, view_spec))
        return voxel_access

    @profile
    def forward(self, input):
        camera, view_spec, ray_spec = input
        # world = VoxelGrid.from_tensor(self.parameter_voxels)
        # Use self.voxels as the weights, take camera as input
        renderer = Renderer(self.parameter_world, camera, view_spec, ray_spec)
        num_stochastic_rays = NUM_STOCHASTIC_RAYS
        self.voxel_access = renderer.build_rays(stochastic_samples(num_stochastic_rays, view_spec))
        r, g, b = renderer.render_from_rays(self.voxel_access)
        modify_grad(self.parameter_world, self.voxel_access)
        return r, g, b, renderer


@profile
def train_minibatch(model, optimizer, camera, view_spec, ray_spec, image_channels, batch_index, epoch_index):
    # This just loads training images and shows them
    # t = transforms.Compose([transforms.ToTensor()])
    # dataset = datasets.ImageFolder("./images", transform=t)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    # images, labels = next(iter(data_loader))
    # training_image = images[0]
    print(f"Shape = {image_channels.shape}")

    optimizer.zero_grad()
    r, g, b, renderer = model([camera, view_spec, ray_spec])
    image = samples_to_image(r, g, b, view_spec)
    renderer.plot_from_image(image, plt, f"Epoch: {epoch_index} Image: {batch_index}")

    red_mse = mse(r, image_channels[0], view_spec)
    green_mse = mse(g, image_channels[1], view_spec)
    blue_mse = mse(b, image_channels[2], view_spec)
    # total_loss = red_mse + green_mse + blue_mse
    print(f"Regularising using {int(len(model.voxel_access.all_voxels) * REGULARISATION_FRACTION)} voxels...")
    total_loss = red_mse + green_mse + blue_mse + REGULARISATION_LAMBDA * tv_term(model.voxel_access,
                                                                                  model.parameter_world)
    print(f"Loss={total_loss}, RGB MSE={(red_mse, green_mse, blue_mse)}")
    total_loss.backward()
    print(f"Model parameters: {len(list(model.parameters()))}")
    # for param in model.parameters():
    #     print(f"Param after={param.grad.shape}")
    #     print(f"Gradients={torch.max(param.grad)}")
    # make_dot(total_mse, params=dict(list(model.named_parameters()))).render("mse", format="png")
    # make_dot(r, params=dict(list(model.named_parameters()))).render("channel", format="png")
    optimizer.step()
    # after = torch.stack(list(model.parameters()))

    return total_loss.detach()


def render_training_images(camera_positions, focal_length, camera_look_at, world, view_spec, ray_spec, plt):
    for index, p in enumerate(camera_positions):
        c = Camera(focal_length, p, camera_look_at)
        r = Renderer(world, c, view_spec, ray_spec)
        red, green, blue = r.render(plt)
        save_image(torch.stack([red, green, blue]), f"./images/training/rotating-cube-{index:02}.png")

    plt.show()
    print("Completed rendering images")


def train(camera_look_at, focal_length, ray_spec, renderer, training_positions, view_spec, world, num_epochs):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.ImageFolder("./images", transform=to_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    training_images = list(data_loader)[0][0]
    model = PlenoxelModel(world)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    epoch_losses = []
    for epoch in range(num_epochs):
        batch_losses = []
        print(f"In epoch {epoch}")
        for batch, position in enumerate(training_positions):
            print(f"Before Training for camera position #{batch}={position}")
            test_camera = Camera(focal_length, position, camera_look_at)
            minibatch_loss = train_minibatch(model, optimizer, test_camera, view_spec, ray_spec,
                                             training_images[batch], batch, epoch)
            batch_losses.append(minibatch_loss)
            print(f"After Training for camera position #{batch}={position}")
        epoch_losses.append(batch_losses)
    red, green, blue = renderer.render(plt)
    transforms.ToPILImage()(torch.stack([red, green, blue])).show()
    print("Rendered final result")
    plt.show()
    return world, epoch_losses


# Reconstructs the world from disk
def reconstruct_flyby_from_file(filename, camera_positions, focal_length, look_at, view_spec, ray_spec):
    voxel_grid = torch.load(filename)
    print(voxel_grid.shape)
    reconstructed_world = VoxelGrid.from_tensor(voxel_grid)
    print("Constructing flyby...")
    reconstruct_flyby_from_world(reconstructed_world, camera_positions, focal_length, look_at, view_spec, ray_spec)


def reconstruct_flyby_from_world(world, camera_positions, focal_length, look_at, view_spec, ray_spec):
    print("Constructing flyby...")
    for index, view_point in enumerate(camera_positions):
        c = Camera(focal_length, view_point, look_at)
        r1 = Renderer(world, c, torch.tensor(view_spec), ray_spec)
        red, green, blue = r1.render(plt, f"Frame {index}")
        save_image(torch.stack([red, green, blue]), f"./random-images/frames/animated-cube-{index:02}.png")
    print("Finished constructing flyby!!")


def main():
    random_world = VoxelGrid.build_random_world(GRID_X, GRID_Y, GRID_Z)
    mono_world = VoxelGrid.build_with_voxel(GRID_X, GRID_Y, GRID_Z, torch.cat(
        [torch.tensor([0.0002, random.random() * 100.]), torch.zeros(VoxelGrid.VOXEL_DIMENSION - 2)]))
    empty_world = VoxelGrid.build_empty_world(GRID_X, GRID_Y, GRID_Z)
    empty_world.build_hollow_cube_with_randomly_coloured_sides(Voxel.uniform_harmonic_random_colour(requires_grad=True),
                                                               torch.tensor([10, 10, 10, 20, 20, 20]))
    world = random_world
    proxy_world = VoxelGrid.new(2, 2, 2, Voxel.default_voxel())
    # empty_world.build_solid_cube(torch.tensor([10, 10, 10, 20, 20, 20]))
    # world.build_monochrome_hollow_cube(torch.tensor([10, 10, 10, 20, 20, 20]))
    cube_center = torch.tensor([20., 20., 20., 1.])
    # camera_look_at = torch.tensor([0., 0., 0., 1])
    camera_look_at = cube_center

    # Exact diagonal centering of cube
    # camera_center = torch.tensor([40., 40., 40., 1.])
    camera_center = torch.tensor([-20., -10., 40., 1.])
    camera_radius = 35.
    # camera_center = torch.tensor([-10.3109, 20.0000, 2.5000, 1.0000])
    focal_length = 1.
    camera = Camera(focal_length, camera_center, camera_look_at)
    num_rays_x, num_rays_y = 50, 50
    view_x1, view_x2 = -1, 1
    view_y1, view_y2 = -1, 1
    view_spec = [view_x1, view_x2, view_y1, view_y2, num_rays_x, num_rays_y]
    ray_length = 100
    num_ray_samples = 100
    ray_spec = torch.tensor([ray_length, num_ray_samples])

    renderer = Renderer(world, camera, torch.tensor(view_spec), ray_spec)

    # This renders the volumetric model and shows the rendered image. Useful for training
    # red, green, blue = r.render(plt)
    # image_tensor = torch.stack([red, green, blue])
    # image = transforms.ToPILImage()(image_tensor)
    # image.show()

    # This draws stochastic rays and returns a set of samples with colours
    num_stochastic_rays = 1000
    # image_data = samples_to_image(r, g, b, view_spec)
    # transforms.ToPILImage()(image_data).show()

    # This draws stochastic rays and returns a set of samples with colours
    # However, it separates out the determining the intersecting voxels and the transmittance
    # calculations, so that it can be put through a Plenoxel model optimisation
    # start_build_rays = timer()
    # # voxel_access = renderer.build_rays(fullscreen_samples(view_spec))
    # voxel_access = renderer.build_rays(stochastic_samples(2000, view_spec))
    # end_build_rays = timer()
    # print(f"Building rays took {end_build_rays - start_build_rays}")
    #
    # start_render_rays = timer()
    # r, g, b = renderer.render_from_rays(voxel_access)
    # end_render_rays = timer()
    # print(f"Rendering rays took {end_render_rays - start_render_rays}")
    #
    # image_data = samples_to_image(r, g, b, view_spec)
    # renderer.plot_from_image(image_data, plt)
    # transforms.ToPILImage()(image_data).show()
    # start_render_full = timer()
    # renderer.render(plt)
    # end_render_full = timer()
    # print(f"Rendering rays in full took {end_render_full - start_render_full}")
    # print("Finished rendering!!")

    # image_data = samples_to_image(r, g, b, view_spec)
    # transforms.ToPILImage()(image_data).show()
    # print(voxel_pointers)
    # print("Render complete")

    # print(f"No of total total ray samples benchmark={len(MASTER_RAY_SAMPLE_POSITIONS_STRUCTURE)}")
    # print(f"No of total total ray samples current={len(voxel_access.ray_sample_positions)}")
    # print(f"Difference={(torch.tensor(MASTER_RAY_SAMPLE_POSITIONS_STRUCTURE) - torch.tensor(voxel_access.ray_sample_positions)).pow(2).sum()}")
    # print(f"Total voxels benchmark={len(MASTER_VOXELS_STRUCTURE)}")
    # print(f"Total voxels current={len(voxel_access.all_voxels)}")
    # print(f"Voxels not used={VOXELS_NOT_USED}")

    # red_mse = mse(r, image[0], view_spec, num_rays_x, num_rays_y)
    # green_mse = mse(g, image[1], view_spec, num_rays_x, num_rays_y)
    # blue_mse = mse(b, image[2], view_spec, num_rays_x, num_rays_y)
    # print(f"{red_mse}, {green_mse}, {blue_mse}")

    # red, green, blue = r.render(plt)
    # transforms.ToPILImage()(torch.stack([red, green, blue])).show()

    # voxel_access, voxels, losses = training_loop(random_world, camera, view_spec, ray_spec, 3)
    # print("Optimisation complete!")
    # update_world(voxels, voxel_access, random_world)
    # red, green, blue = r.render(plt)
    # transforms.ToPILImage()(torch.stack([red, green, blue])).show()
    # print("Rendered final result")

    # Generates training images
    # camera_positions = generate_camera_angles(camera_radius, cube_center)
    # render_training_images(camera_positions, focal_length, cube_center, world, view_spec, ray_spec, plt, camera_radius)

    RECONSTRUCTED_WORLD_FILENAME = "reconstructed.pt"
    # Trains on multiple training images
    test_positions = torch.tensor([[-20., -10., 40., 1.]])
    training_positions = torch.tensor([[-4.7487, 44.7487, 20.0000, 1.0000],
                                       [-3.9054, -3.9054, 29.0587, 1.0000],
                                       [-1.4330, 41.4330, 2.5000, 1.0000],
                                       [-1.4330, 41.4330, 37.5000, 1.0000],
                                       [2.5000, 2.5000, -4.7487, 1.0000],
                                       [7.6256, 32.3744, -10.3109, 1.0000],
                                       [13.5946, 13.5946, 53.8074, 1.0000],
                                       [20.0000, 20.0000, -15.0000, 1.0000],
                                       [20.0000, 20.0000, 55.0000, 1.0000],
                                       [26.4054, 26.4054, -13.8074, 1.0000],
                                       [32.3744, 7.6256, -10.3109, 1.0000],
                                       [37.5000, 37.5000, 44.7487, 1.0000],
                                       [43.9054, 43.9054, 10.9413, 1.0000],
                                       [44.7487, -4.7487, 20.0000, 1.0000]])

    num_epochs = 20
    reconstructed_world, epoch_losses = train(camera_look_at, focal_length, ray_spec, renderer, training_positions,
                                              view_spec, world, num_epochs)
    print(f"Epoch losses = {epoch_losses}")
    torch.save(reconstructed_world.voxel_grid, RECONSTRUCTED_WORLD_FILENAME)
    print(f"Saved world to {RECONSTRUCTED_WORLD_FILENAME}!")
    reconstruct_flyby_from_file(RECONSTRUCTED_WORLD_FILENAME, training_positions, focal_length, camera_look_at,
                                view_spec,
                                ray_spec)
    camera_positions = generate_camera_angles(camera_radius, cube_center)
    # reconstruct_flyby_from_world(empty_world, training_positions, focal_length, camera_look_at, view_spec,
    #                              ray_spec)
    print("Everything done!!")


if __name__ == '__main__':
    main()
