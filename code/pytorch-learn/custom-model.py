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


def render_image(world):
    test_voxel = world[0, 0, 0]

    proxy_intersecting_voxels = torch.stack([test_voxel])
    proxy_red_channel = torch.stack([torch.stack([torch.tensor(0), torch.tensor(0), test_voxel[2] * 5])])
    proxy_green_channel = torch.stack([torch.stack([torch.tensor(0), torch.tensor(0), test_voxel[2] * 5])])
    proxy_blue_channel = torch.stack([torch.stack([torch.tensor(0), torch.tensor(0), test_voxel[2] * 5])])
    return (proxy_red_channel, proxy_green_channel, proxy_blue_channel, proxy_intersecting_voxels)


# print(render_image(voxel_grid))

world = torch.rand([2, 2], requires_grad=True)


# world *=4
# world += 10

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.initial = torch.rand([2, 2], requires_grad=True)
        self.initial = world
        # self.parameter = nn.Parameter(self.initial)
        print("Rendered image")
        # r, g, b, intersecting = render_image(voxel_grid)
        self.voxels = nn.Parameter(voxels)

    def forward(self, x):
        result = torch.stack([torch.stack([self.parameter[0, 0], self.parameter[0, 0] * 2]),
                              torch.stack([self.parameter[0, 0], self.parameter[0, 0] * 2])])
        result2 = self.parameter[1, 0] + self.parameter[1, 0]
        # print(result)
        return result, result2


net = CustomModel()
print(net)
print(list(net.parameters()))

learning_rate = 0.0001
simple_optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
target = torch.tensor(200.)
loss_criterion = nn.MSELoss()


# net.train()

def custom_loss(output, target):
    return (output - target).pow(2)


print(list(net.parameters()))
for i in range(1):
    simple_optimiser.zero_grad()
    # output = net(torch.tensor(20.))
    output1, output2 = net([])
    make_dot(net.initial, params=dict(list(net.named_parameters()))).render("custom_model", format="png")
    # print(f"Output={output}")

    print(f"output1={output1}")
    loss1 = output1[0, 0] - 200
    loss2 = custom_loss(output2, 200)
    print(f"Loss = {loss1}, {loss2}")
    for p in net.parameters():
        print(f"Before={p.grad}")

    loss3 = loss1 + loss2
    loss3.backward()
    for p in net.parameters():
        print("AHA")
        print(f"After={p.grad}")
    simple_optimiser.step()

# print(list(net.parameters()))
net.eval()
print(net(torch.tensor(1.)))
