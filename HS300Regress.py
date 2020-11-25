import pandas as pd
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

start_time = time.time()

# 数据集加载
data = pd.read_csv('SZ#399300.csv', header=0)
X = data.values[:, 1:]
y = data['Close'].values
y = np.diff(y, prepend=y[0])
X = np.array(X - np.min(X, axis=0) / (np.max(X, axis=0) -
                                      np.min(X, axis=0)), dtype=np.float64)

# 超参数设置
lr = 0.01
epoch = 20
input_dim = X.shape[1]
output_dim = 1
test_size = 0.5
seed = 27
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 划分训练集和测试集合
bound = int(y.shape[0]*(1-test_size))
X_train = torch.FloatTensor(X[:bound].reshape(-1, 1, input_dim)).cuda()
y_train = torch.FloatTensor(y[:bound]).cuda()
X_test = torch.FloatTensor(X[bound:].reshape(-1, 1, input_dim)).cuda()
y_test = torch.FloatTensor(y[bound:]).cuda()


class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layer)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.out(out[:, -1, :])
        return out


def plot(loss, y, out):
    sns.set()
    plt.figure('loss')
    plt.plot(loss[0], label='loss_train')
    plt.plot(loss[0], label='loss_test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.figure('predict')
    plt.plot(y, label='truth')
    plt.plot(out, label='predict')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('predict.png')


# 设置网络，优化器，损失函数
net = Net(input_size=input_dim, hidden_size=32,
          output_size=output_dim, num_layer=8)
net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss_func = nn.MSELoss()

# 训练数据
loss_np = np.zeros((2, epoch))
for i in range(epoch):
    out = net(X_train)
    out = out.reshape(out.shape[0])
    loss = loss_func(out, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    out_test = net(X_test)
    out_test = out_test.reshape(out_test.shape[0])
    loss_test = loss_func(out_test, y_test)
    if ((i+1) % 1 == 0):
        print("epoch %3d" % (i+1),
              "train loss: {:.4f} test loss: {:.4f}".format(loss, loss_test))
    loss_np[0, i] = loss
    loss_np[1, i] = loss_test

end_time = time.time()

print("loss: {:.4f} time:{:.2f}s".format(loss_test, end_time-start_time))
loss_np = loss_np
y_test = y_test.cpu()
out_test = out_test.cpu().detach().numpy()
plot(loss_np, y_test, out_test)
