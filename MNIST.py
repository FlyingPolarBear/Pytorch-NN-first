import torch
import torch.nn as nn
from torchvision import datasets, transforms

# 超参数设置
BATCH_SIZE = 512
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 下载训练集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1037,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)

# 测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1*1*28*28
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)  # 1* 10 * 24 *24
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2, 2)  # 1* 10 * 12 * 12
        x = self.conv2(x)  # 1* 20 * 10 * 10
        x = nn.functional.relu(x)
        x = x.view(in_size, -1)  # 1 * 2000
        x = self.fc1(x)  # 1 * 500
        x = nn.functional.relu(x)
        x = self.fc2(x)  # 1 * 10
        x = nn.functional.log_softmax(x, dim=1)
        return x


model = ConvNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())


def train(model, device, train_loader, optimizer, epoch):  # 定义训练函数
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):  # 定义测试函数
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target,
                                                reduction='sum')  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


# 最后开始训练和测试
for epoch in range(1, EPOCHS+1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
