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
data = load_iris()
X = data.data
y = data.target
X = np.array(X - np.min(X, axis=0)) / \
    (np.max(X, axis=0)-np.min(X, axis=0))
y = np.array(y)

# 超参数设置
lr = 0.01
epoch = 200
input_dim = X.shape[1]
output_dim = 3
test_size = 0.3
seed = 27

# 划分训练集和测试集合
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed)

X_train = torch.FloatTensor(X_train.reshape(-1, 1, input_dim))
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test.reshape(-1, 1, input_dim))
y_test = torch.LongTensor(y_test)


class NET(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, output_size=3, num_layer=2):
        super(NET, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layer)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.out(out[:, -1, :])
        return out

    def evaluate(self, x, y):
        out = self.forward(x)
        poss = nn.functional.softmax(out, dim=1)
        y_pred = torch.max(poss, 1)[1]
        accuracy = ((y_pred == y).int().sum().float() /
                    float(y.shape[0])).numpy()
        return accuracy, y_pred

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
net = NET(input_size=4, hidden_size=16, output_size=3, num_layer=2)
optimizer = torch.optim.Adam(net.parameters(), lr=0.08, betas=(
    0.9, 0.999), eps=1e-08, weight_decay=0)
loss_func = torch.nn.CrossEntropyLoss()

# 训练数据
acc_np = np.zeros(epoch)
loss_np = np.zeros(epoch)
for i in range(epoch):
    out = net(X_train)
    loss = loss_func(out, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc, y_pred = net.evaluate(X_test, y_test)
    if ((i+1) % 10 == 0):
        print("epoch %4d" % (i+1), "loss: {:.4f} acc: {:.4f}".format(loss, acc))
    acc_np[i] = acc
    loss_np[i] = loss

end_time = time.time()
acc, y_pred = net.evaluate(X_test, y_test)
print("acc:", acc, "time:{:.2f}s".format(end_time-start_time))
plot(loss_np, acc_np)
