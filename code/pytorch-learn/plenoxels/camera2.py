import math

import numpy as np
import torch
import matplotlib.pyplot as plt


class Camera:
    def __init__(self, focal_length, center, basis):
        camera_center = center.detach().clone()
        transposed_basis = torch.transpose(basis, 0, 1)
        camera_center[:3] = camera_center[
                            :3] * -1  # We don't want to multiply the homogenous coordinate component; it needs to remain 1
        camera_origin_translation = torch.eye(4, 4)
        camera_origin_translation[:, 3] = camera_center
        extrinsic_camera_parameters = torch.matmul(torch.inverse(transposed_basis), camera_origin_translation)
        intrinsic_camera_parameters = torch.tensor([[focal_length, 0., 0., 0.],
                                                    [0., focal_length, 0., 0.],
                                                    [0., 0., 1., 0.],
                                                    [0., 0., 0., 1.]])
        self.transform = torch.matmul(intrinsic_camera_parameters, extrinsic_camera_parameters)

    def to_2D(self, point):
        rendered_point = torch.matmul(self.transform, torch.transpose(point, 0, 1))
        point_z = rendered_point[2, 0]
        return rendered_point / point_z


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


look_at = torch.tensor([0., 0., 0., 1])
camera_center = torch.tensor([-5., -10., 20., 1.])
focal_length = 1.

camera_basis = basis_from_depth(look_at, camera_center)
camera = Camera(focal_length, camera_center, camera_basis)

fig1 = plt.figure()

for i in range(10):
    for j in range(10):
        for k in range(10):
            d = camera.to_2D(torch.tensor([[i, j, k, 1.]]))
            plt.plot(d[0][0], d[1][0], marker="o")

ray_origin = camera_center
camera_basis_x = camera_basis[0][:3]
camera_basis_y = camera_basis[1][:3]

unit_vector_x_camera_basis = unit_vector(camera_basis_x)
unit_vector_y_camera_basis = unit_vector(camera_basis_y)

print(unit_vector_x_camera_basis)
print(unit_vector_y_camera_basis)

camera_center_inhomogenous = camera_center[:3]

fig2 = plt.figure()

for i in np.linspace(-10, 20, 50):
    for j in np.linspace(0, 30, 50):
        ray_screen_intersection = unit_vector_x_camera_basis * i + unit_vector_y_camera_basis * j
        unit_ray = unit_vector(ray_screen_intersection - camera_center_inhomogenous)
        density = 0.
        for k in np.linspace(0, 100):
            ray_endpoint = camera_center_inhomogenous + unit_ray * k
            ray_x, ray_y, ray_z = ray_endpoint
            if (ray_x < 0 or ray_x > 10 or
                    ray_y < 0 or ray_y > 10 or
                    ray_z < 0 or ray_z > 10):
                continue
            # We are in the box
            density += 0.1
        plt.plot(i, j, marker="o", color=str(1. - density))

plt.show()
