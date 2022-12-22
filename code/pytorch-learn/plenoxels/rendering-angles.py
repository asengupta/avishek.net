import functools
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchviz import make_dot

radius = 40.
cube_center = torch.tensor([20., 20., 20., 1.])

camera_positions = []
for phi in np.linspace(0, 2 * math.pi, 4):
    for theta in np.linspace(0, 2 * math.pi, 4):
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        camera_positions.append(torch.tensor([x, y, z, 0]))


print((torch.stack(camera_positions) + cube_center).unique(dim=0))
# print(torch.tensor(camera_positions))
# print((torch.tensor(camera_positions) + cube_center).unique(dim=0))
