import functools
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchviz import make_dot

RED_CHANNEL = 0
GREEN_CHANNEL = 0
BLUE_CHANNEL = 0


class Camera:
    def __init__(self, focal_length, center, basis):
        self.basis = basis
        camera_center = center.detach().clone()
        transposed_basis = torch.transpose(basis, 0, 1)
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
        camera_basis_theta = math.atan(camera_basis_z[1] / camera_basis_z[2])
        camera_basis_phi = math.atan(camera_basis_z[2] / camera_basis_z.norm())
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


def line(marker="o"):
    return lambda p1, p2: plt.plot([p1[0][0], p2[0][0]], [p1[1][0], p2[1][0]], marker="o")


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
    @staticmethod
    def random_voxel():
        # voxel = list(map(lambda x: torch.tensor(random.random(), requires_grad=True), range(VoxelGrid.VOXEL_DIMENSION)))
        # voxel[0] = 0.5
        voxel = torch.cat([torch.tensor([0.005]), torch.rand(VoxelGrid.VOXEL_DIMENSION - 1)])
        voxel.requires_grad = True
        return voxel

    @staticmethod
    def default_voxel():
        voxel = torch.cat([torch.tensor([0.005]), torch.ones(VoxelGrid.VOXEL_DIMENSION - 1)])
        voxel.requires_grad = True
        return voxel

    @staticmethod
    def empty_voxel():
        return torch.zeros([VoxelGrid.VOXEL_DIMENSION], requires_grad=True)


class VoxelGrid:
    VOXEL_DIMENSION = 28

    def __init__(self, x, y, z):
        self.grid_x = x
        self.grid_y = y
        self.grid_z = z
        self.training_voxel = lambda: torch.ones(VoxelGrid.VOXEL_DIMENSION, requires_grad=True)
        # self.voxel_grid = torch.zeros([self.grid_x, self.grid_y, self.grid_z, VoxelGrid.VOXEL_DIMENSION],
        #                              requires_grad=True)
        self.voxel_grid = np.ndarray((self.grid_x, self.grid_y, self.grid_z), dtype=list)
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                for k in range(self.grid_z):
                    self.voxel_grid[i, j, k] = Voxel.random_voxel()
                    # print(f"({i},{j},{k})")

    def at(self, x, y, z):
        if self.is_outside(x, y, z):
            return Voxel.empty_voxel()
        else:
            return self.voxel_grid[int(x), int(y), int(z)]

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
            # print(base_transmittance)
            # if (base_transmittance > 1. or base_transmittance < 0.):
            #     raise Exception(f"Transmittance was {base_transmittance}")
            # if (r > 1. or r < 0. or g > 1. or g < 0. or b > 1. or b < 0.):
            #     raise Exception(f"Harmonic values were ({r}, {g}, {b})")

            # Stacking instead of cat-ing to preserve gradient
            color_densities += torch.stack([base_transmittance * r, base_transmittance * g, base_transmittance * b])

        return color_densities

    def build_solid_cube(self):
        for i in range(13, self.grid_x - 13):
            for j in range(13, self.grid_y - 13):
                for k in range(13, self.grid_z - 13):
                    self.voxel_grid[i, j, k] = Voxel.default_voxel()

    def build_random_hollow_cube(self):
        self.build_hollow_cube(Voxel.random_voxel)

    def build_monochrome_hollow_cube(self, cube_spec):
        self.build_random_hollow_cube2(Voxel.default_voxel, cube_spec)

    def build_hollow_cube(self, make_voxel):
        self.build_random_hollow_cube2(make_voxel, torch.tensor([10, 10, 10, 20, 20, 20]))

    def build_random_hollow_cube2(self, make_voxel, cube_spec):
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

    def density(self, ray_samples_with_distances, viewing_angle):
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

    return ([c000, c001, c010, c011, c100, c101, c110, c111], [(x0, y0, z0),
                                                               (x0, y0, z1),
                                                               (x0, y1, z0),
                                                               (x0, y1, z1),
                                                               (x1, y0, z0),
                                                               (x1, y0, z1),
                                                               (x1, y1, z0),
                                                               (x1, y1, z1)])


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

    # Break up render_image() into two parts:
    # 1) One part will simply create ray samples hashed by rays. This will be used in init() of the custom model
    # 2) The second part will calculate the (r,g,b) triplet and will be used in the forward() pass of the custom model
    def render_image(self, num_stochastic_samples, plt, requires_grad=False):
        # test_voxel = self.world.at(0, 0, 0)
        #
        # proxy_intersecting_voxels = torch.stack([test_voxel])
        # proxy_red_channel = torch.stack([torch.stack([torch.tensor(0), torch.tensor(0), test_voxel[5] * 5])])
        # proxy_green_channel = torch.stack([torch.stack([torch.tensor(0), torch.tensor(0), test_voxel[5] * 5])])
        # proxy_blue_channel = torch.stack([torch.stack([torch.tensor(0), torch.tensor(0), test_voxel[5] * 5])])
        # return (proxy_red_channel, proxy_green_channel, proxy_blue_channel, proxy_intersecting_voxels)
        camera_basis_x = camera.basis[0][:3]
        camera_basis_y = camera.basis[1][:3]
        viewing_angle = camera.viewing_angle()
        camera_center_inhomogenous = camera_center[:3]
        view_length = self.x_2 - self.x_1
        view_height = self.y_2 - self.y_1
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0
        plt.figure(frameon=False)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.figure()
        plt.axis("equal")
        plt.style.use("dark_background")
        ax = plt.gca()
        ax.set_aspect('equal')

        # Need to convert the range [Random(0,1), Random(0,1)] into bounds of [[x1, x2], [y1, y2]]
        ray_intersection_weights = list(
            map(lambda x: torch.mul(torch.rand(2), torch.tensor([view_length, view_height])) + torch.tensor(
                [self.x_1, self.y_1]), list(range(0, num_stochastic_samples))))
        red_channel = []
        green_channel = []
        blue_channel = []

        intersecting_voxels = []
        for ray_intersection_weight in ray_intersection_weights:
            ray_screen_intersection = camera_basis_x * ray_intersection_weight[0] + \
                                      camera_basis_y * ray_intersection_weight[1]
            unit_ray = unit_vector(ray_screen_intersection - camera_center_inhomogenous)
            view_x, view_y = ray_intersection_weight[0], ray_intersection_weight[1]
            ray_samples = []
            for k in np.linspace(0, self.ray_length, self.num_ray_samples):
                ray_endpoint = camera_center_inhomogenous + unit_ray * k
                ray_x, ray_y, ray_z = ray_endpoint
                if (self.world.is_outside(ray_x, ray_y, ray_z)):
                    continue
                # We are in the box
                ray_samples.append([ray_x, ray_y, ray_z])
                # intersecting_voxels.append(torch.stack([ray_x, ray_y, ray_z]))
                at = world.at(ray_x, ray_y, ray_z)
                # print(at.requires_grad)
                intersecting_voxels.append(torch.cat([torch.stack([ray_x, ray_y, ray_z]), at]))

            unique_ray_samples = torch.unique(torch.tensor(ray_samples), dim=0)
            if (len(unique_ray_samples) <= 1):
                red_channel.append(torch.tensor([view_x, view_y, 0.]))
                green_channel.append(torch.tensor([view_x, view_y, 0.]))
                blue_channel.append(torch.tensor([view_x, view_y, 0.]))
                plt.plot(ray_intersection_weight[0], ray_intersection_weight[1], marker="o", color=[0, 0, 0])
                continue
            t1 = unique_ray_samples[:-1]
            t2 = unique_ray_samples[1:]
            consecutive_sample_distances = (t1 - t2).pow(2).sum(1).sqrt()

            # Make 1D tensor into 2D tensor
            ray_samples_with_distances = torch.cat([t1, torch.reshape(consecutive_sample_distances, (-1, 1))], 1)
            color_densities = self.world.density(ray_samples_with_distances, viewing_angle)
            # print(color_densities)

            color_tensor = 1. - torch.clamp(color_densities, min=0, max=1)
            plt.plot(ray_intersection_weight[0], ray_intersection_weight[1], marker="o",
                     color=color_tensor.detach().numpy())

            if (view_x < view_x1 or view_x > view_x2
                    or view_y < view_y1 or view_y > view_y2):
                print(f"Warning: bad generation: {view_x}, {view_y}")

            # print(color_tensor)
            red_channel.append(torch.stack([view_x, view_y, color_tensor[RED_CHANNEL]]))
            green_channel.append(torch.stack([view_x, view_y, color_tensor[GREEN_CHANNEL]]))
            blue_channel.append(torch.stack([view_x, view_y, color_tensor[BLUE_CHANNEL]]))

        # Remember to flip to prevent image being rendered upside down when saved to a file
        plt.show()
        print("Done!!")
        intersecting_voxels = torch.stack(intersecting_voxels)
        red_channel, green_channel, blue_channel = torch.stack(red_channel), torch.stack(green_channel), torch.stack(
            blue_channel)
        return (red_channel, green_channel, blue_channel, intersecting_voxels)

    def render_from_rays(self, view_points, voxel_positions_by_rays, voxel_pointers, all_voxels, plt):
        viewing_angle = camera.viewing_angle()
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0
        plt.figure(frameon=False)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.figure()
        plt.axis("equal")
        plt.style.use("dark_background")
        ax = plt.gca()
        ax.set_aspect('equal')

        # Need to convert the range [Random(0,1), Random(0,1)] into bounds of [[x1, x2], [y1, y2]]
        red_channel = []
        green_channel = []
        blue_channel = []

        for ray_index, voxels_pointer in enumerate(voxel_pointers):
            voxels_in_ray = all_voxels[voxels_pointer[0]: voxels_pointer[1]]
            voxel_positions = voxel_positions_by_rays[ray_index]
            unique_ray_samples = torch.unique(torch.tensor(voxel_positions), dim=0)
            view_x, view_y = view_points[ray_index]
            if (len(unique_ray_samples) <= 1):
                red_channel.append(torch.tensor([view_x, view_y, 0.]))
                green_channel.append(torch.tensor([view_x, view_y, 0.]))
                blue_channel.append(torch.tensor([view_x, view_y, 0.]))
                plt.plot(view_x, view_y, marker="o", color=[0, 0, 0])
                continue
            t1 = unique_ray_samples[:-1]
            t2 = unique_ray_samples[1:]
            consecutive_sample_distances = (t1 - t2).pow(2).sum(1).sqrt()

            # Make 1D tensor into 2D tensor
            # List of tensors, last element of each tensor is distance to the next tensor
            ray_samples_with_distances = torch.cat([t1, torch.reshape(consecutive_sample_distances, (-1, 1))], 1)
            color_densities = self.world.density(ray_samples_with_distances, viewing_angle)
            color_tensor = 1. - torch.clamp(color_densities, min=0, max=1)
            plt.plot(view_x, view_y, marker="o",
                     color=color_tensor.detach().numpy())

            if (view_x < view_x1 or view_x > view_x2
                    or view_y < view_y1 or view_y > view_y2):
                print(f"Warning: bad generation: {view_x}, {view_y}")

            # print(color_tensor)
            red_channel.append(torch.stack([view_x, view_y, color_tensor[RED_CHANNEL]]))
            green_channel.append(torch.stack([view_x, view_y, color_tensor[GREEN_CHANNEL]]))
            blue_channel.append(torch.stack([view_x, view_y, color_tensor[BLUE_CHANNEL]]))

        plt.show()
        print("Done!!")
        red_channel, green_channel, blue_channel = torch.stack(red_channel), torch.stack(green_channel), torch.stack(
            blue_channel)
        return (red_channel, green_channel, blue_channel)

    def build_rays(self, num_stochastic_samples):
        camera_basis_x = camera.basis[0][:3]
        camera_basis_y = camera.basis[1][:3]
        camera_center_inhomogenous = camera_center[:3]
        view_length = self.x_2 - self.x_1
        view_height = self.y_2 - self.y_1

        # Need to convert the range [Random(0,1), Random(0,1)] into bounds of [[x1, x2], [y1, y2]]
        ray_intersection_weights = list(
            map(lambda x: torch.mul(torch.rand(2), torch.tensor([view_length, view_height])) + torch.tensor(
                [self.x_1, self.y_1]), list(range(0, num_stochastic_samples))))
        voxels_by_rays = []
        voxel_positions_by_rays = []
        view_points = []
        voxel_pointers = []
        all_voxels = []
        counter = 0
        for ray_intersection_weight in ray_intersection_weights:
            ray_screen_intersection = camera_basis_x * ray_intersection_weight[0] + \
                                      camera_basis_y * ray_intersection_weight[1]
            unit_ray = unit_vector(ray_screen_intersection - camera_center_inhomogenous)
            view_x, view_y = ray_intersection_weight[0], ray_intersection_weight[1]
            voxels_per_ray = []
            voxel_positions_per_ray = []
            for k in np.linspace(0, self.ray_length, self.num_ray_samples):
                ray_endpoint = camera_center_inhomogenous + unit_ray * k
                ray_x, ray_y, ray_z = ray_endpoint
                if (self.world.is_outside(ray_x, ray_y, ray_z)):
                    continue
                # We are in the box
                at = world.at(ray_x, ray_y, ray_z)
                interpolating_voxels, interpolating_voxel_positions = neighbours(world, ray_x, ray_y, ray_z)
                voxels_per_ray.append(at)
                all_voxels.append(at)
                # voxels_per_ray.append(torch.cat([torch.stack([ray_x, ray_y, ray_z]), at]))
                voxel_positions_per_ray.append([ray_x, ray_y, ray_z])
            if (not voxels_per_ray):
                continue
            voxels_by_rays.append(torch.stack(voxels_per_ray))
            view_points.append((view_x, view_y))
            voxel_positions_by_rays.append(voxel_positions_per_ray)
            voxel_pointers.append((counter, counter + len(voxels_per_ray)))
            counter += len(voxels_per_ray)

            if (view_x < view_x1 or view_x > view_x2
                    or view_y < view_y1 or view_y > view_y2):
                print(f"Warning: bad generation: {view_x}, {view_y}")
        print("Done!!")

        return view_points, voxel_positions_by_rays, voxels_by_rays, voxel_pointers, all_voxels

    def render(self, plt):
        red_image = []
        green_image = []
        blue_image = []
        camera_basis_x = camera.basis[0][:3]
        camera_basis_y = camera.basis[1][:3]
        viewing_angle = camera.viewing_angle()
        camera_center_inhomogenous = camera_center[:3]

        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0
        plt.figure(frameon=False)
        plt.rcParams['axes.facecolor'] = 'black'
        plt.axis("equal")
        plt.figure()
        plt.style.use("dark_background")
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.axis("off")
        for i in np.linspace(self.x_1, self.x_2, self.num_view_samples_x):
            red_column = []
            green_column = []
            blue_column = []
            for j in np.linspace(self.y_1, self.y_2, self.num_view_samples_y):
                ray_screen_intersection = camera_basis_x * i + camera_basis_y * j
                unit_ray = unit_vector(ray_screen_intersection - camera_center_inhomogenous)
                ray_samples = []
                # To remove artifacts, set ray step samples to be higher, like 200
                for k in np.linspace(0, self.ray_length, self.num_ray_samples):
                    ray_endpoint = camera_center_inhomogenous + unit_ray * k
                    ray_x, ray_y, ray_z = ray_endpoint
                    if (self.world.is_outside(ray_x, ray_y, ray_z)):
                        continue
                    # We are in the box
                    ray_samples.append([ray_x, ray_y, ray_z])

                unique_ray_samples = torch.unique(torch.tensor(ray_samples), dim=0)
                if (len(unique_ray_samples) <= 1):
                    red_column.append(torch.tensor(0))
                    green_column.append(torch.tensor(0))
                    blue_column.append(torch.tensor(0))
                    plt.plot(i, j, marker="o", color=[0, 0, 0])
                    continue
                t1 = unique_ray_samples[:-1]
                t2 = unique_ray_samples[1:]
                consecutive_sample_distances = (t1 - t2).pow(2).sum(1).sqrt()

                # Make 1D tensor into 2D tensor
                ray_samples_with_distances = torch.cat([t1, torch.reshape(consecutive_sample_distances, (-1, 1))], 1)
                color_densities = self.world.density(ray_samples_with_distances, viewing_angle)
                print(color_densities)

                color_tensor = 1. - torch.clamp(color_densities, min=0, max=1)
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
        plt.show()
        print("Done!!")
        return (red_image_tensor, green_image_tensor, blue_image_tensor)


def camera_to_image(x, y, view_spec):
    view_x1, view_x2, view_y1, view_y2, num_rays_x, num_rays_y = view_spec
    step_x = (view_x2 - view_x1) / num_rays_x
    step_y = (view_y2 - view_y1) / num_rays_y
    return (int((x - view_x1) / step_x), int((view_y2 - y) / step_y))


def samples_to_image(red_samples, green_samples, blue_samples, view_spec):
    X, Y, INTENSITY = 0, 1, 2
    view_x1, view_x2, view_y1, view_y2, num_rays_x, num_rays_y = view_spec
    step_x = (view_x2 - view_x1) / num_rays_x
    step_y = (view_y2 - view_y1) / num_rays_y
    red_render_channel = torch.zeros([num_rays_y, num_rays_x])
    green_render_channel = torch.zeros([num_rays_y, num_rays_x])
    blue_render_channel = torch.zeros([num_rays_y, num_rays_x])
    for index, pixel in enumerate(red_samples):
        print(
            f"({view_y2 - pixel[1]}, {pixel[0] - view_x1}) -> ({int((view_y2 - pixel[1]) / step_y)}, {int((pixel[0] - view_x1) / step_x)}), {pixel[2]}")
        # red_render_channel[int((view_y2 - pixel[1]) / step_y), int((pixel[0] - view_x1) / step_x)] = red_samples[index][
        #     2]
        # green_render_channel[int((view_y2 - pixel[1]) / step_y), int((pixel[0] - view_x1) / step_x)] = \
        # green_samples[index][2]
        # blue_render_channel[int((view_y2 - pixel[1]) / step_y), int((pixel[0] - view_x1) / step_x)] = \
        # blue_samples[index][2]
        x, y = camera_to_image(pixel[X], pixel[Y], view_spec)
        red_render_channel[y, x] = red_samples[index][INTENSITY]
        green_render_channel[y, x] = green_samples[index][INTENSITY]
        blue_render_channel[y, x] = blue_samples[index][INTENSITY]
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


NUM_STOCHASTIC_RAYS = 200


class PlenoxelModel(nn.Module):
    def __init__(self, input):
        super().__init__()
        camera, view_spec, ray_spec = input
        self.world = world
        view_points, voxel_positions_by_rays, voxel_pointers, all_voxels = PlenoxelModel.run(world, [camera, view_spec,
                                                                                                     ray_spec])
        self.parameters = nn.Parameter(torch.stack(all_voxels), requires_grad=True)
        self.view_points = view_points
        self.voxel_positions_by_rays = voxel_positions_by_rays
        self.voxel_pointers = voxel_pointers
        self.all_voxels = all_voxels

    @staticmethod
    def run(world, input):
        camera, view_spec, ray_spec = input
        renderer = Renderer(world, camera, view_spec, ray_spec)
        # This draws stochastic rays and returns a set of samples with colours
        num_stochastic_rays = NUM_STOCHASTIC_RAYS
        view_points, voxel_positions_by_rays, voxels_by_rays, voxel_pointers, all_voxels = renderer.build_rays(
            num_stochastic_rays)
        return view_points, voxel_positions_by_rays, voxel_pointers, all_voxels

    def forward(self, input):
        camera, view_spec, ray_spec = input
        # Use self.voxels as the weights, take camera as input
        renderer = Renderer(world, camera, view_spec, ray_spec)
        # This just loads training images and shows them
        # t = transforms.Compose([transforms.ToTensor()])
        # dataset = datasets.ImageFolder("./images", transform=t)
        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        # images, labels = next(iter(data_loader))
        # image = images[0]
        # transforms.ToPILImage()(image).show()

        # This draws stochastic rays and returns a set of samples with colours
        num_stochastic_rays = NUM_STOCHASTIC_RAYS
        all_voxels = self.parameters
        r, g, b = renderer.render_from_rays(self.view_points, self.voxel_positions_by_rays, self.voxel_pointers,
                                            all_voxels, plt)
        # r, g, b, voxels = renderer.render_image(num_stochastic_rays, plt, requires_grad=True)
        # image_data = samples_to_image(r, g, b, view_spec)
        # transforms.ToPILImage()(image_data).show()
        return r, g, b
        # green_mse = mse(g, image[1], view_spec, num_rays_x, num_rays_y)
        # blue_mse = mse(b, image[2], view_spec, num_rays_x, num_rays_y)
        # return red_mse


def training_loop(world, camera, view_spec, ray_spec, n=1):
    losses = []
    # This just loads training images and shows them
    t = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.ImageFolder("./images", transform=t)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    images, labels = next(iter(data_loader))
    training_image = images[0]
    print(f"{n} epochs")

    for i in range(n):
        print(f"Epoch={i}")
        model = PlenoxelModel([camera, view_spec, ray_spec])
        # print(list(model.parameters()))
        # r, g, b, voxels = model([camera, view_spec, ray_spec])
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, momentum=0.9)
        print(list(model.parameters()))
        optimizer.zero_grad()
        r, g, b = model([camera, view_spec, ray_spec])

        red_mse = mse(r, training_image[0], view_spec)
        green_mse = mse(g, training_image[0], view_spec)
        blue_mse = mse(b, training_image[0], view_spec)
        total_mse = red_mse + green_mse + blue_mse
        print(f"MSE={total_mse}")
        # print(f"Optimising {len(voxels)} voxels...")
        # before = voxels.clone()
        # # print(optimizer.param_groups)
        # print("Backward")
        total_mse.backward()
        for param in model.parameters():
            print(f"Param after={param.grad}")
        make_dot(total_mse, params=dict(list(model.named_parameters()))).render("mse", format="png")
        # make_dot(r, params=dict(list(model.named_parameters()))).render("channel", format="png")
        # print(f"shape={r.shape}")
        # print(r)
        # print(total_mse)
        # print(world.at(0, 0, 0))
        # print(list(model.parameters()))
        # for param in model.parameters():
        #     print(f"Param before={param}")
        optimizer.step()
        # after = torch.stack(list(model.parameters()))
        # print((after - before).abs().sum())
        # weight_change = (before - voxels).abs().sum()
        # print(f"Weight change={weight_change}")
        # losses.append(total_mse)
    return losses


GRID_X = 40
GRID_Y = 40
GRID_Z = 40

world = VoxelGrid(GRID_X, GRID_Y, GRID_Z)
proxy_world = VoxelGrid(2, 2, 2)
# world.build_solid_cube()
# world.build_random_hollow_cube()
# world.build_monochrome_hollow_cube(torch.tensor([10, 10, 10, 20, 20, 20]))
world.build_random_hollow_cube2(Voxel.default_voxel, torch.tensor([10, 10, 10, 20, 20, 20]))
# world.build_random_hollow_cube2(Voxel.random_voxel, torch.tensor([15, 15, 15, 10, 10, 10]))

camera_look_at = torch.tensor([0., 0., 0., 1])
# camera_center = torch.tensor([-60., 5., 15., 1.])
# camera_center = torch.tensor([-10., -10., 15., 1.])
camera_center = torch.tensor([-20., -10., 40., 1.])
# camera_center = torch.tensor([-20., -20., 40., 1.])
focal_length = 1.

camera_basis = basis_from_depth(camera_look_at, camera_center)
camera = Camera(focal_length, camera_center, camera_basis)
num_rays_x = 100
num_rays_y = 100
view_x1 = -35
view_x2 = 30
view_y1 = -15
view_y2 = 60
view_spec = [view_x1, view_x2, view_y1, view_y2, num_rays_x, num_rays_y]
ray_spec = torch.tensor([100, 100])
r = Renderer(world, camera, torch.tensor([view_x1, view_x2, view_y1, view_y2, num_rays_x, num_rays_y]),
             ray_spec)

# This renders the volumetric model and shows the rendered image. Useful for training
# red, green, blue = r.render(plt)
# print(red.shape)
# print(green.shape)
# print(blue.shape)
# transforms.ToPILImage()(torch.stack([red, green, blue])).show()

# This just loads training images and shows them
# t = transforms.Compose([transforms.ToTensor()])
# dataset = datasets.ImageFolder("./images", transform=t)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
# images, labels = next(iter(data_loader))
# image = images[0]
# transforms.ToPILImage()(image).show()

# This draws stochastic rays and returns a set of samples with colours
num_stochastic_rays = 2000
# r, g, b, intersecting_voxels = r.render_image(num_stochastic_rays, plt)
# image_data = samples_to_image(r, g, b, view_spec)
# transforms.ToPILImage()(image_data).show()

# This draws stochastic rays and returns a set of samples with colours
# However, it separates out the determining the intersecting voxels and the transmittance
# calculations, so that it can be put through a Plenoxel model optimisation
# view_points, voxel_positions_by_rays, voxels_by_rays, voxel_pointers, all_voxels = r.build_rays(num_stochastic_rays)
# r, g, b = r.render_from_rays(view_points, voxel_positions_by_rays, voxel_pointers, all_voxels, plt)
# image_data = samples_to_image(r, g, b, view_spec)
# transforms.ToPILImage()(image_data).show()
# print(voxel_pointers)
# print("Render complete")

# red_mse = mse(r, image[0], view_spec, num_rays_x, num_rays_y)
# green_mse = mse(g, image[1], view_spec, num_rays_x, num_rays_y)
# blue_mse = mse(b, image[2], view_spec, num_rays_x, num_rays_y)
# print(f"{red_mse}, {green_mse}, {blue_mse}")

# red, green, blue = r.render(plt)
# transforms.ToPILImage()(torch.stack([red, green, blue])).show()

training_loop(world, camera, view_spec, ray_spec, 1)
# print("Optimisation complete!")
# red, green, blue = r.render(plt)
# transforms.ToPILImage()(torch.stack([red, green, blue])).show()

# Calculates MSE against whole images
# total_num_rays = num_rays_x * num_rays_y
# red_error = (red - image[0]).pow(2).sum() / total_num_rays
# green_error = (green - image[1]).pow(2).sum() / total_num_rays
# blue_error = (blue - image[2]).pow(2).sum() / total_num_rays
#
# print(red_error)
# print(green_error)
# print(blue_error)

# x = np.array([[1, 2], [3, 4, 5], [6, 7, 8, 9]], dtype=object)
# counter = 0
# pointers = []
# for i in x:
#     pointers.append((counter, counter + len(i)))
#     counter += len(i)
#
# print(pointers)
# x = np.concatenate(x).ravel()
# print(x)
# print(pointers[2][0])
# print(x[pointers[2][0]: pointers[2][1]])
