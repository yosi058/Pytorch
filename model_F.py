import sys

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchvision import datasets


class FirstNet(nn.Module):
    def __init__(self, image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        # x = F.sigmoid(self.fc5(x))
        return F.log_softmax(x, dim=1)


def train(model, epochs=10):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def test():
    model.eval()
    test_loss = 0
    correct = 0
    print("Test")
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).cpu().sum()
        test_loss /= (len(test_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def start_model():
    for e in range(1, 11):
        train(model, e)
        test()


if __name__ == '__main__':
    x1, y1, z1 = sys.argv[1], sys.argv[2], sys.argv[3]
    train_x = np.loadtxt(x1)
    train_y = np.loadtxt(y1)
    test_x = np.loadtxt(z1)
    train_x = train_x / 255
    test_x = test_x / 255

    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).long()
    dataset = TensorDataset(train_x, train_y)
    train_set, validate_set = torch.utils.data.random_split(dataset, [44000, 11000])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(validate_set, batch_size=32, shuffle=True)

    # transforms = transforms.Compose([transforms.ToTensor(),
    #                                  transforms.Normalize((0.1307,), (0.3081,))])
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms), batch_size=64,
    #     shuffle=True)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST(root='./data', train=False, transform=transforms), batch_size=64, shuffle=True)

    model = FirstNet(image_size=28 * 28)
    lr = 0.05
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # train_x, train_y, test_x, test_y = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    # my_model = Model(train_x, train_y, test_x, test_y)
    start_model()
