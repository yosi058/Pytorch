import sys

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt


class FirstNet(nn.Module):
    def __init__(self, image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 1000)
        self.fc1 = nn.Linear(1000, 600)
        self.fc2 = nn.Linear(600, 250)
        self.fc3 = nn.Linear(250, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        norm = nn.BatchNorm1d(1000)
        x = F.leaky_relu(norm(self.fc0(x)))
        norm = nn.BatchNorm1d(600)
        x = F.leaky_relu(norm(self.fc1(x)))
        norm = nn.BatchNorm1d(250)
        x = F.leaky_relu(norm(self.fc2(x)))
        norm = nn.BatchNorm1d(50)
        x = F.leaky_relu(norm(self.fc3(x)))
        x = F.leaky_relu(self.fc4(x))
        return F.log_softmax(x, dim=1)


def train(model, epochs=10):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        train_loss += F.nll_loss(output, labels, size_average=False).item()
        loss.backward()
        optimizer.step()
    return train_loss / len(train_loader.dataset)


# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0
#     print("Test")
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = model(data)
#             test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
#             pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).cpu().sum()
#         test_loss /= (len(test_loader.dataset))
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#             test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
#     return test_loss


def start_model():
    for e in range(0, 25):
        print("the number of epoch is", e)
        if e % 5 == 0 and e > 1:
            optimizer.param_groups[0]['lr'] *= 0.1

        train(model, e)


if __name__ == '__main__':
    x1, y1, z1, test_y = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    train_x = np.loadtxt(x1)
    train_y = np.loadtxt(y1)
    test_x = np.loadtxt(z1)
    train_x = train_x / 255
    test_x = test_x / 255
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).long()
    dataset = TensorDataset(train_x, train_y)
    # train_set = torch.utils.data.random_split(dataset)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    # test_loader = DataLoader(validate_set, batch_size=128, shuffle=True)

    # transforms = transforms.Compose([transforms.ToTensor(),
    #                                  transforms.Normalize((0.1307,), (0.3081,))])
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms), batch_size=128,
    #     shuffle=True)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST(root='./data', train=False, transform=transforms), batch_size=128, shuffle=True)

    model = FirstNet(image_size=28 * 28)
    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_model()
    test_y = open("test_y", "w")
    print("start to write...")
    for t in test_x:
        output = model(t)
        y_hat = output.max(1, keepdim=True)[1]
        test_y.write(str(y_hat) + "\n")
    test_y.close()
