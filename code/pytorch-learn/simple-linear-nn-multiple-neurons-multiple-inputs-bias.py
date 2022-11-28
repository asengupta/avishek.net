import torch
import torch.nn.functional as F
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.first_layer = nn.Linear(2, 3, bias=True)

    def forward(self, x):
        return F.leaky_relu(self.first_layer(x))


net = SimpleNN()
print(net)
print(list(net.parameters()))
net.train()

learning_rate = 0.1
simple_optimiser = torch.optim.SGD(net.parameters(), lr = learning_rate, momentum=0.9)
target =torch.tensor([5., 6., 7.])
loss_criterion =nn.MSELoss()

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
print(F.leaky_relu(torch.tensor([-1., 5.])))
