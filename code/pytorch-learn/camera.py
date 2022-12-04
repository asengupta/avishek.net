import math

import torch

point0 = torch.transpose(torch.tensor([[10., 10., 0., 1.]]), 0, 1)
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


print(torch.matmul(translate(20, 30, 40), point1))
print(torch.matmul(rotation_z_axis(math.pi/4), point2))

focal_length = 10.

camera_coordinate_system = torch.transpose(torch.tensor([[1.,1.,0.,0.], [-1.,1.,0,0.], [.0,0.,1.,0.], [0.,0.,0.,1.]]),0,1)
# camera_coordinate_system = torch.transpose(torch.tensor([[math.cos(math.pi/4),math.sin(math.pi/4),0.,0.], [-math.cos(math.pi/4),math.sin(math.pi/4),0,0.], [.0,0.,1.,0.], [0.,0.,0.,1.]]),0,1)
camera_center = torch.tensor([5.,5.,0.,1.])
camera_center[:3] = camera_center[:3] * -1
print(camera_center)
camera_origin_translation = torch.eye(4, 4)
camera_origin_translation[:, 3] = camera_center
print(camera_origin_translation)
change_of_basis_to_camera = torch.matmul(torch.inverse(camera_coordinate_system), camera_origin_translation)
print(change_of_basis_to_camera)
point_in_new_basis = torch.matmul(change_of_basis_to_camera, point0)
print(point_in_new_basis)

intrinsic_camera_parameters = torch.tensor([[]])
