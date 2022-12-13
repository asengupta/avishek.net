import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        initial = torch.rand([2, 2, 2], requires_grad=True)
        self.parameter = nn.Parameter(initial)

    def forward(self, x):
        result = self.parameter[0, 0, 0] + self.parameter[0, 0, 1]
        # print(result)
        return result


net = CustomModel()
print(net)
print(list(net.parameters()))

learning_rate = 0.0001
simple_optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
target = torch.tensor(200.)
loss_criterion = nn.MSELoss()
net.train()

print(list(net.parameters()))
for i in range(1000):
    simple_optimiser.zero_grad()
    # output = net(torch.tensor(20.))
    output = net([])
    # print(f"Output={output}")

    loss = loss_criterion(output, target)
    print(f"Loss = {loss}")
    for p in net.parameters():
        print(f"Before={p.grad}")
    print(loss)

    loss.backward()
    for p in net.parameters():
        print(f"After={p.grad}")
    simple_optimiser.step()

print(list(net.parameters()))
net.eval()
print(net(torch.tensor(1.)))
