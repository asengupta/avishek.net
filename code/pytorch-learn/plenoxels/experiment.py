import torch
import math
import functools

number_of_samples = 5
distance_density_color_tensors = torch.tensor([
    [1., 1.],
    [2., 2.],
    [.3, .3],
    [4., 4.],
    [5., 5.]
])

colors = torch.tensor([0.5] * 5)
transmittances = list(map(lambda i: functools.reduce(
    lambda acc, j: acc + distance_density_color_tensors[j, 1] * distance_density_color_tensors[j, 0],
    range(0, i), 0.), range(1, number_of_samples + 1)))
transmittances = torch.exp(- torch.stack(transmittances))

print(transmittances)
# print(distance_density_color_tensors[:, 0] * distance_density_color_tensors[:, 1])

sigma_density = distance_density_color_tensors[:, 0] * distance_density_color_tensors[:, 1]
summing_matrix = torch.tensor(list(
    functools.reduce(lambda acc, n: acc + [[1.] * n + [0.] * (number_of_samples - n)], range(1, number_of_samples + 1),
                     [])))
# print(result.t())
new_transmittances = torch.matmul(sigma_density, summing_matrix.t())
print(torch.exp(-new_transmittances))

new_transmittances * (1 -torch.exp(sigma_density)) * colors
