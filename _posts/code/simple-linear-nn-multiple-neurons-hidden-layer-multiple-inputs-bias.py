import torch
import torch.nn.functional as F
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.first_layer = nn.Linear(2, 3, bias=True)
        self.second_layer = nn.Linear(3, 1, bias=True)

    def forward(self, x):
        return F.leaky_relu(self.second_layer(self.first_layer(x)))


net = SimpleNN()
print(net)
print(list(net.parameters()))
net.train()

learning_rate = 0.01
simple_optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
target = torch.tensor([10.])
loss_criterion = nn.MSELoss()

for i in range(100):
    simple_optimiser.zero_grad()
    output = net(torch.tensor([1., 1.]))

    loss = loss_criterion(output, target)
    print(f"Loss = {loss}")

    loss.backward()
    simple_optimiser.step()

print(list(net.parameters()))
net.eval()
print(net(torch.tensor([1., 1.])))
