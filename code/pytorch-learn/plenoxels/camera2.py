import math

import torch
import matplotlib.pyplot as plt

point0 = torch.transpose(torch.tensor([[10., 10., 15., 1.]]), 0, 1)
point3 = torch.transpose(torch.tensor([[30., 30., 15., 1.]]), 0, 1)
point1 = torch.transpose(torch.tensor([[1, 1, 1, 1]]), 0, 1)
point2 = torch.transpose(torch.tensor([[1., 1., 0., 1.]]), 0, 1)


def rotation_x_axis(angle):
    return torch.tensor([[1, 0, 0, 0],
                         [0, math.cos(angle), - math.sin(angle), 0],
                         [0, math.sin(angle), math.cos(angle), 0],
                         [0, 0, 0, 1]])


def rotation_y_axis(angle):
    return torch.tensor([[1, 0, 0, 0],
                         [math.cos(angle), 0, math.sin(angle), 0],
                         [0, 1, 0, 0],
                         [0, -math.sin(angle), 0, math.cos(angle)],
                         [0, 0, 0, 1]])


def rotation_z_axis(angle):
    return torch.tensor([[math.cos(angle), - math.sin(angle), 0, 5],
                         [math.sin(angle), math.cos(angle), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])


def translate(x, y, z):
    return torch.tensor([[0, 0, 0, x],
                         [0, 0, 0, y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]])


focal_length = 1.


class Camera:
    def __init__(self, focal_length, center, basis):
        camera_center = center.detach().clone()
        transposed_basis = torch.transpose(basis, 0, 1)
        camera_center[:3] = camera_center[:3] * -1
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
        # return rendered_point
        point_z = rendered_point[2, 0]
        return rendered_point / point_z


def camera_basis_from(camera_depth_z_vector):
    depth_vector = camera_depth_z_vector[:3]
    cartesian_z_vector = torch.tensor([0., 0., 1.])
    cartesian_z_projection_lambda = torch.dot(depth_vector, cartesian_z_vector) / torch.dot(
        depth_vector, depth_vector)
    camera_up_vector = cartesian_z_vector - cartesian_z_projection_lambda * depth_vector
    camera_x_vector = torch.linalg.cross(depth_vector, camera_up_vector)
    inhomogeneous_basis = torch.stack([camera_x_vector, camera_up_vector, depth_vector, torch.tensor([0., 0., 0.])])
    homogeneous_basis = torch.hstack((inhomogeneous_basis, torch.tensor([[0.], [0.], [0.], [1.]])))
    return homogeneous_basis


def basis_from_depth(look_at, camera_center):
    depth_vector = torch.sub(look_at, camera_center)
    depth_vector[3] = 1.
    return camera_basis_from(depth_vector)


look_at2 = torch.tensor([0., 0., 0., 1])
camera_center2 = torch.tensor([-10., -10., 20., 1.])

camera2 = Camera(focal_length, camera_center2, basis_from_depth(look_at2, camera_center2))

def plot(style="bo"):
    return lambda p: plt.plot(p[0][0], p[1][0], style)


def line(style="bo"):
    return lambda p1, p2: plt.plot([p1[0][0], p2[0][0]], [p1[1][0], p2[1][0]], marker="o")

for i in range(10):
    for j in range(10):
        for k in range(10):
            print(i)
            d = camera2.to_2D(torch.tensor([[i, j, k, 1.]]))
            plt.plot(d[0][0], d[1][0], marker="o")

plt.show()

start_ray = camera_center2

