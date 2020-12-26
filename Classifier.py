import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def LoadData():
    X, y = load_iris(return_X_y=True)  # 加载数据集
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)  # 归一化
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=27)  # 划分训练集和测试集
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
    dataset_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=32, shuffle=True)
    return dataset_train, X_test, y_test


# 模型定义
class Network(nn.Module):
    def __init__(self, n_in, n_out, n_hid1=32, n_hid2=16):
        super().__init__()
        self.hidden1 = nn.Linear(n_in, n_hid1)
        self.hidden2 = nn.Linear(n_hid1, n_hid2)
        self.output = nn.Linear(n_hid2, n_out)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.dropout1(x)
        x = F.relu(self.hidden2(x))
        x = self.dropout2(x)
        x = F.relu(self.output(x))
        return x

    def evaluate(self, x, y):
        out = self.forward(x)
        poss = F.softmax(out, dim=1)
        y_pred = torch.max(poss, 1)[1]
        accuracy = ((y_pred == y).int().sum().float() /
                    float(y.shape[0])).numpy()
        return accuracy


def Plot(loss, acc):
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


def Train(dataset_train, X_test, y_test):
    epoch = 2000
    model = Network(n_in=4, n_out=3)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    acc_all = np.zeros(epoch)
    loss_all = np.zeros(epoch)
    for i in range(epoch):
        optimizer.zero_grad()  # 梯度清零
        for X, y in dataset_train:
            y_pred = model(X)  # 前向传播
            loss = loss_func(y_pred, y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 梯度下降
        acc = model.evaluate(X_test, y_test)
        acc_all[i] = acc  # 记录损失
        loss_all[i] = loss  # 记录损失
        if (i+1) % 100 == 0:
            print("epoch {:4d} loss: {:.2f} acc: {:.2f}%".format(
                i+1, loss, 100*acc))
    return model, loss_all, acc_all


def Pred(X_test, y_test, model):
    acc = model.evaluate(X_test, y_test)
    print("acc: {:.2f}%".format(100*acc))


if __name__ == "__main__":
    dataset_train, X_test, y_test = LoadData()
    model, loss_all, acc_all = Train(dataset_train, X_test, y_test)
    Pred(X_test, y_test, model)
    Plot(loss_all, acc_all)
