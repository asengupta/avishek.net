import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        initial = torch.rand([2, 2, 2], requires_grad=True)
        self.parameter = nn.Parameter(initial)

    def forward(self, x):
        return self.parameter[0,0,0] + self.parameter[0,0,1]


net = CustomModel()
print(net)
print(list(net.parameters()))

learning_rate = 0.0001
simple_optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
target = torch.tensor(200.)
loss_criterion = nn.MSELoss()
net.train()

for i in range(5000):
    simple_optimiser.zero_grad()
    output = net(torch.tensor(20.))
    # print(f"Output={output}")

    loss = loss_criterion(output, target)
    print(f"Loss = {loss}")
    for p in net.parameters():
        print(p)
    # print(loss)

    loss.backward()
    simple_optimiser.step()

print(list(net.parameters()))
net.eval()
print(net(torch.tensor(1.)))
