import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
import torch.nn as nn

data = [[1, 2], [3, 4]]
data_tensor = torch.tensor(data)
print(data_tensor)
data_tensor_2 = torch.clone(data_tensor)
print(data_tensor_2)
data_tensor[0, 0] = 2
print(data_tensor)
print(data_tensor_2)

shape_2d = (2, 3, 4)
print(shape_2d[1])
ones = torch.ones(shape_2d)
print(ones)

print(ones.shape)
print(ones.dtype)
print(ones.device)

ones[0, 1] = torch.tensor([3, 4, 5, 6])
ones[0, :, 3] = torch.tensor([7, 8, 9])
ones[0, :2, 2:] = torch.tensor([[20, 21], [22, 23]])
print(ones)
print(ones.mul(ones))

weights = torch.tensor([[2, 7], [3, 8], [4, 9]])
x = torch.tensor([[2], [3], [4]])
print(torch.adjoint(weights).matmul(x))

model = resnet18(weights=ResNet18_Weights.DEFAULT)
print(model)
input = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
prediction = model(input)
print(prediction.shape)
print(labels.shape)
loss = (prediction - labels).sum()
loss.backward()

optimiser = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimiser.step()


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.first_layer = nn.Linear(2, 1, bias=True)

    def forward(self, x):
        print(f"Input={x}")
        return F.relu(self.first_layer(x))


net = SimpleNN()
learning_rate = 0.001
simple_optimiser = torch.optim.SGD(net.parameters(), lr = learning_rate, momentum=0.9)
target =torch.tensor([5.])
criterion =nn.MSELoss()

print(net)
print(list(net.parameters()))

for i in range(100):
    simple_optimiser.zero_grad()
    output = net(torch.tensor([1., 1.]))
    # print(output)

    loss = criterion(output, target)
    print(f"Loss = {loss}")

    loss.backward()
    simple_optimiser.step()

    # print(list(net.parameters()))

print(list(net.parameters()))
