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
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


def train(model, epochs=10):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        train_loss += F.nll_loss(output, labels,size_average=False).item()
        loss.backward()
        optimizer.step()
    return train_loss / len(train_loader.dataset)


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
    return test_loss


def start_model():
    train_loss = []
    test_loss = []
    for e in range(1,11):
        train_loss.append(float(train(model, e)))
        test_loss.append(float(test()))
    return train_loss,test_loss

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

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(validate_set, batch_size=128, shuffle=True)

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
    train_loss_arr,validate_loss_arr = start_model()
    print(train_loss_arr)
    print(list(range(3)))
    plt.plot(list(range(10)),train_loss_arr, label= "line_train")
    plt.plot(list(range(10)),validate_loss_arr,label="test")
    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title(f"k=")
    plt.show()
