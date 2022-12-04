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


# print(torch.matmul(translate(20, 30, 40), point1))
# print(torch.matmul(rotation_z_axis(math.pi/4), point2))

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

# camera_coordinate_system = torch.tensor([[1.,1.,0.,0.], [-1.,1.,0,0.], [.0,0.,1.,0.], [0.,0.,0.,1.]])
# camera_basis = torch.tensor([[1.,-1.,0.,0.], [0.,0.,1.,0.], [1.,1.,0.,0.], [0.,0.,0.,1.]])
camera_basis = torch.tensor([[1.,-1.,0.,0.], [1.,1.,1.,0.], [1.,1.,-2.,0.], [0.,0.,0.,1.]])
# camera_basis = torch.tensor([[1.,-1.,0.,0.], [1.,1.,1.,0.], [1.,1.,-1.,0.], [0.,0.,0.,1.]])
# camera_coordinate_system = torch.tensor([[math.cos(math.pi/4),math.sin(math.pi/4),0.,0.], [-math.cos(math.pi/4),math.sin(math.pi/4),0,0.], [.0,0.,1.,0.], [0.,0.,0.,1.]])
camera_center = torch.tensor([0.,0.,10.,1.])

camera = Camera(focal_length, camera_center, camera_basis)
r1 = camera.to_2D(torch.tensor([[10., 10., 10., 1.]]))
r2 = camera.to_2D(torch.tensor([[30., 30., 10., 1.]]))

front_zx = 10
front_zy = 10

back_zx = 17
back_zy = 17
front_sq_1 = camera.to_2D(torch.tensor([[front_zx - 5., front_zy + 5., 5., 1.]]))
front_sq_2 = camera.to_2D(torch.tensor([[front_zx - 5., front_zy + 5., -5., 1.]]))
front_sq_3 = camera.to_2D(torch.tensor([[front_zx + 5., front_zy - 5., 5., 1.]]))
front_sq_4 = camera.to_2D(torch.tensor([[front_zx + 5., front_zy - 5., -5., 1.]]))

far = 50
back_sq_1 = camera.to_2D(torch.tensor([[back_zx - 5., back_zy + 5., 5., 1.]]))
back_sq_2 = camera.to_2D(torch.tensor([[back_zx - 5., back_zy + 5., -5., 1.]]))
back_sq_3 = camera.to_2D(torch.tensor([[back_zx + 5., back_zy - 5., 5., 1.]]))
back_sq_4 = camera.to_2D(torch.tensor([[back_zx + 5., back_zy - 5., -5., 1.]]))

print(front_sq_1)
print(front_sq_2)
print(front_sq_3)
print(front_sq_4)

def plot(style="bo"):
    return lambda p: plt.plot(p[0][0], p[1][0], style)

def line(style="bo"):
    return lambda p1, p2: plt.plot([p1[0][0], p2[0][0]], [p1[1][0], p2[1][0]], marker="o")

front_line = line("bo")
back_line = line("ro")

front_line(front_sq_1, front_sq_2)
front_line(front_sq_2, front_sq_4)
front_line(front_sq_3, front_sq_4)
front_line(front_sq_3, front_sq_1)

back_line(back_sq_1, back_sq_2)
back_line(back_sq_2, back_sq_4)
back_line(back_sq_3, back_sq_4)
back_line(back_sq_3, back_sq_1)

plt.show()
