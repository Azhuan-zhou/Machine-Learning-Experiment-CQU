import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import torch.utils.data
from torch.utils.data import Dataset

plt.switch_backend('TkAgg')


# logistic 函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x.sum()))


def loss(label, x, w):
    return -label * np.dot(w[0].T, x) + np.log(1 + np.exp(np.dot(w[0].T, x)))


def gradient(output, label, x):
    return (-label + output) * x


def plot_loss(loss, epochs):
    fig, ax = plt.subplots()
    ax.plot(epochs, loss)
    ax.set_title('loss')
    ax.set(xlabel='epoch', ylabel='loss')
    plt.show()
    plt.close()

# 训练模型
def trainLogRegres(train_x, train_y, opts):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    # 计算训练时间
    startTime = time.time()
    # 样本个数与样本特征个数，注意x=(1,x1,x2),是一个增广矩阵
    numSamples, numFeatures = np.shape(train_x)
    # 学习率
    alpha = opts['alpha']
    # 最大迭代次数
    maxIter = opts['maxIter']
    # 权重初始化[1,1，1]
    weights = np.ones((1, numFeatures))
    losses = []
    # 随机梯度下降算法
    for epoch in range(maxIter):
        dataIndex = [i for i in range(numSamples)]
        loss_average = 0
        for i in range(numSamples):
            alpha = 4.0 / (1.0 + epoch + i) + 0.01
            # 随机获取一个样本的标签
            Index = int(np.random.uniform(0, len(dataIndex)))
            randIndex = dataIndex[Index]
            # 输入sigmoid，获得输出值
            output = sigmoid(train_x[randIndex, :] * weights)
            # 计算损失：
            error = loss(train_y[randIndex], train_x[randIndex, :], weights)
            weights = weights - alpha * gradient(output, train_y[randIndex], train_x[randIndex])
            del (dataIndex[Index])  # 删除已经训练过的样本
            loss_average += error
        print('epoch:{}/{},loss:{}'.format(epoch + 1, maxIter, loss_average / numSamples))
        losses.append(loss_average / numSamples)
    plot_loss(losses, range(maxIter))
    print(' training complete! Took {}'.format(time.time() - startTime))
    return weights


# 测试模型
def testLogRegres(weights, test_x, test_y):
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    numSamples, numFeatures = np.shape(test_x)
    matchCount = 0
    for i in range(numSamples):
        predict = sigmoid(test_x[i, :] * weights) > 0.5
        if predict == bool(test_y[i]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy


# 显示logistic模型
def showLogRegres(weights, train_x, train_y):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    # 训练样本个数，训练样本的特征数目
    numSamples, numFeatures = np.shape(train_x)
    # draw all samples
    for i in range(numSamples):
        if int(train_y[i]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

    # 画出超平面
    min_x = min(train_x[:, 1])
    max_x = max(train_x[:, 1])
    y_min_x = float(-weights[0, 0] - weights[0, 1] * min_x) / weights[0, 2]
    y_max_x = float(-weights[0, 0] - weights[0, 1] * max_x) / weights[0, 2]

    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Result')
    plt.show()


class MyDataSet(Dataset):
    def __getitem__(self, index):
        return self.dataset[index]

    def __init__(self):
        data = pd.read_excel('test.xls')
        dataset = []
        for i in range(100):
            dataset.append(data.loc[i].to_list())
        dataset = np.array(dataset)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)


def loadData(dataset):
    trainSet, testSet = torch.utils.data.random_split(dataset, [70, 30])
    return trainSet, testSet


# 加载模型
print("step 1: load data...")
dataset = MyDataSet()
trainSet, testSet = loadData(dataset)
# 加载训练集数据
train_features = []
train_labels = []
for train_data in trainSet:
    train_features.append([1, train_data[0], train_data[1]])
    train_labels.append(train_data[2])

# 训练模型
print("step 2: training...")
opts = {'alpha': 0.01, 'maxIter': 100, 'optimizeType': 'smoothStocGradDescent'}
# 加载测试集数据
optimalWeights = trainLogRegres(train_features, train_labels, opts)
test_features = []
test_labels = []
for test_data in testSet:
    test_features.append([1, test_data[0], test_data[1]])
    test_labels.append(test_data[2])
# 训练
print("step 3: testing...")
accuracy = testLogRegres(optimalWeights, test_features, test_labels)

# 显示
print("step 4: show the result...")
print('The classify accuracy is: {}%'.format(accuracy * 100))
showLogRegres(optimalWeights, train_features, train_labels)
