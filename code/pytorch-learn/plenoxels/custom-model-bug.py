import torch
import torch.nn as nn
from torchviz import make_dot
import numpy as np


# world = [torch.tensor(2., requires_grad=True), torch.tensor(3., requires_grad=True)]
def random_voxel():
    voxel = torch.cat([torch.tensor([0.005]), torch.ones(2)])
    voxel.requires_grad = True
    return voxel


x = y = z = 1
voxel_grid = np.ndarray((1, 1, 1), dtype=list)
for i in range(x):
    for j in range(y):
        for k in range(z):
            voxel_grid[i, j, k] = random_voxel()


def render_image(world, voxels):
    if (voxels is None):
        test_voxel = world[0, 0, 0]
    else:
        test_voxel = voxels[0]

    proxy_intersecting_voxels = torch.stack([test_voxel])
    proxy_red_channel = torch.stack([torch.stack([torch.tensor(0), torch.tensor(0), test_voxel[2] * 5])])
    proxy_green_channel = torch.stack([torch.stack([torch.tensor(0), torch.tensor(0), test_voxel[2] * 5])])
    proxy_blue_channel = torch.stack([torch.stack([torch.tensor(0), torch.tensor(0), test_voxel[2] * 5])])
    return (proxy_red_channel, proxy_green_channel, proxy_blue_channel, proxy_intersecting_voxels)


# print(render_image(voxel_grid))

world = torch.rand([2, 2], requires_grad=True)


# world *=4
# world += 10
def mse(rendered_channel):
    true_channel = torch.ones([2, 2]) * 10
    channel_total_error = torch.tensor(0.)
    for point in rendered_channel:
        x, y, intensity = point
        intensity = intensity
        image_x, image_y = x, y
        pixel_error = (true_channel[int(image_y), int(image_x)] - intensity).pow(2)
        # print(pixel_error)
        channel_total_error += pixel_error
    return channel_total_error / len(rendered_channel)

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.initial = torch.rand([2, 2], requires_grad=True)
        self.initial = world
        # self.parameter = nn.Parameter(self.initial)
        print("Rendered image")
        r, g, b, intersecting = render_image(voxel_grid, None)
        self.voxels = nn.Parameter(intersecting)

    def forward(self, x):
        r, g, b, intersecting = render_image(voxel_grid, self.voxels)
        return r, g, b, intersecting

model = CustomModel()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, momentum=0.9)
optimizer.zero_grad()
r, g, b, intersecting = model([])
red_mse = mse(r)
green_mse = mse(g)
blue_mse = mse(b)
total_mse = red_mse + green_mse + blue_mse
print(f"MSE=")
print(total_mse)

print(list(model.parameters()))
for param in model.parameters():
    print(f"Param before={param.grad}")
total_mse.backward()
for param in model.parameters():
    print(f"Param after={param.grad}")
