import torch
import torch.nn as nn
from torchviz import make_dot
import numpy as np
from torch.functional import F


class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """

    def __init__(self):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((3,))
        # make weights torch parameters
        self.weights = nn.Parameter(weights)

    def forward(self, X):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-k * X) + b),
        """
        a, k, b = self.weights
        return a * torch.exp(-k * X) + b


def training_loop(model, optimizer, n=1000):
    "Training loop for torch model."
    losses = []
    for i in range(n):
        preds = model(10)
        loss = F.mse_loss(preds, torch.tensor(30.)).sqrt()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)
    return losses


m = Model()
# Instantiate optimizer
opt = torch.optim.Adam(m.parameters(), lr=0.001)
losses = training_loop(m, opt)

print(list(m.parameters()))
for p in m.parameters():
    print(p.grad)

