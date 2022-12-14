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

world = torch.rand([2, 2], requires_grad=True)

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.intersecting = torch.tensor([[0.005, 1., 1.]], requires_grad=True)
        intersecting = torch.tensor([0.005, 1., 1.], requires_grad=True)
        self.voxels = nn.Parameter(intersecting)

    def forward(self, x):
        x1 = self.voxels[0] * x
        # The following line will give zero gradient. Treat the parameter as the actual object itself
        # x1 = self.intersecting[0] * x
        return x1


model = CustomModel()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer.zero_grad()
intersecting = model(2)
print("------------------OUTPUT------------------")
print(intersecting)
red_mse = nn.MSELoss()(intersecting, torch.tensor(500.))
total_mse = red_mse
print(f"MSE=")
print(total_mse)

print(list(model.parameters()))
for param in model.parameters():
    print(f"Param before={param.grad}")
total_mse.backward()
for param in model.parameters():
    print(f"Param after={param.grad}")

# net = CustomModel()
# print(net)
# print(list(net.parameters()))
#
# learning_rate = 0.0001
# simple_optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# target = torch.tensor(200.)
# loss_criterion = nn.MSELoss()
#
#
# # net.train()
#
# def custom_loss(output, target):
#     return (output - target).pow(2)
#
#
# print(list(net.parameters()))
# for i in range(1):
#     simple_optimiser.zero_grad()
#     # output = net(torch.tensor(20.))
#     output1, output2 = net([])
#     make_dot(net.initial, params=dict(list(net.named_parameters()))).render("custom_model", format="png")
#     # print(f"Output={output}")
#
#     print(f"output1={output1}")
#     loss1 = output1[0, 0] - 200
#     loss2 = custom_loss(output2, 200)
#     print(f"Loss = {loss1}, {loss2}")
#     for p in net.parameters():
#         print(f"Before={p.grad}")
#
#     loss3 = loss1 + loss2
#     loss3.backward()
#     for p in net.parameters():
#         print("AHA")
#         print(f"After={p.grad}")
#     simple_optimiser.step()
#
# # print(list(net.parameters()))
# net.eval()
# print(net(torch.tensor(1.)))
