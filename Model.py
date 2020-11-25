import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

start_time = time.time()

# 数据集加载
dataset = load_iris()
X = dataset.data
y = dataset.target

# 超参数设置
lr = 0.01
epoch = 5000
input_dim = X.shape[1]
output_dim = 3
test_size = 0.7
seed = 27

# 划分训练集和测试集合
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)


class Net(torch.nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_in, n_hid)
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, x):
        x = self.hidden(x)
        x = nn.functional.relu(x)
        x = self.out(x)
        return x

    def evaluate(self, x, y):
        out = self.forward(x)
        poss = nn.functional.softmax(out, dim=1)
        y_pred = torch.max(poss, 1)[1]
        accuracy = ((y_pred == y).int().sum().float() /
                    float(y.shape[0])).numpy()
        return accuracy


def plot(loss, acc):
    sns.set()
    plt.figure('loss')
    plt.plot(loss, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss.png')
    plt.figure('acc')
    plt.plot(acc, label='accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('acc.png')


# 设置网络，优化器，损失函数
net = Net(n_in=input_dim, n_hid=16, n_out=output_dim)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

# 训练数据
acc_np = np.zeros(epoch)
loss_np = np.zeros(epoch)
for i in range(epoch):
    out = net(X_train)
    loss = loss_func(out, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = net.evaluate(X_test, y_test)
    if ((i+1) % 100 == 0):
        print("epoch %4d" % (i+1), "loss: {:.4f} acc: {:.4f}".format(loss, acc))
    acc_np[i] = acc
    loss_np[i] = loss

end_time = time.time()
acc = net.evaluate(X_test, y_test)
print("acc:", acc, "time:{:.2f}s".format(end_time-start_time))
plot(loss_np, acc_np)
