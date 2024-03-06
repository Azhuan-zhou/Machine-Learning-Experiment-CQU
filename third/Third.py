from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import torch
import numpy as np
import time
from tqdm import tqdm
import cv2 as cv

fun = lambda a, b: 1 if a == b else 0


# ----------------搭建数据集-----------------
def DataSet(root):
    transform = transforms.Compose(
        [transforms.Resize((28, 28)),
         transforms.ToTensor()]
    )
    # 加载数据
    training_set = datasets.MNIST(root,
                                  train=True,
                                  transform=transform,
                                  download=False,
                                  )
    testing_set = datasets.MNIST(root,
                                 train=False,
                                 transform=transform,
                                 download=False
                                 )
    subset_train = list(range(6000))
    subset_test = list(range(1000))
    training_set = Subset(training_set, subset_train)
    testing_set = Subset(testing_set, subset_test)
    training_set_loader = DataLoader(training_set,
                                     batch_size=32,
                                     shuffle=True)
    testing_set_loader = DataLoader(testing_set,
                                    batch_size=32,
                                    shuffle=True)
    return training_set_loader, testing_set_loader


def sigmoid(x):
    F = torch.nn.Sigmoid()
    return F(x)


def softmax(x):
    f = torch.nn.Softmax(dim=1)
    return f(x)


def forward(data):
    # 预处理数
    batch_size = data.shape[0]
    data = data.reshape(batch_size, -1)
    inputSize = data.shape[-1]
    # 前馈传播
    # 输入层到隐藏层一
    x1 = torch.matmul(data, w[1])
    a1 = sigmoid(x1)
    # 隐藏层一到隐藏层二
    x2 = torch.matmul(a1, w[2])
    a2 = sigmoid(x2)
    # 隐藏层二到隐藏层三
    x3 = torch.matmul(a2, w[3])
    a3 = sigmoid(x3)
    # 隐藏层三到隐藏层四
    x4 = torch.matmul(a3, w[4])
    a4 = sigmoid(x4)
    # 隐藏层四到输出层
    output = torch.matmul(a4, w[5])
    # 隐藏层的状态
    a = {1: a1, 2: a2, 3: a3, 4: a4}
    return output, a


def CrossEntropy(output, label):
    # softmax
    a = softmax(output)
    loss = []
    for i, j in enumerate(label):
        lo = -torch.log(a[i, j])
        loss.append(lo)
    loss = torch.tensor(loss)
    return loss


# 计算正确率
def accuracy(output, label):
    y = output.argmax(axis=1)
    acc = y == label
    return acc.sum() / len(label)


# 反向传播
def backward(output, data, label, a):
    """
    :param output: 神经网络的输出
    :param data: 神经网络的输入数据
    :param label: 输入数据的标签
    :param a: 每个隐藏层的输入
    :return:
    """
    batch_size = output.shape[0]
    a5 = softmax(output)
    # 保存每次计算的梯度
    w_1 = torch.zeros_like(w[1]).to(device)
    w_2 = torch.zeros_like(w[2]).to(device)
    w_3 = torch.zeros_like(w[3]).to(device)
    w_4 = torch.zeros_like(w[4]).to(device)
    w_5 = torch.zeros_like(w[5]).to(device)
    for i in range(batch_size):
        # 处理数据
        label_list = [fun(k, label[i]) for k in range(10)]  # 热编码
        label_list = torch.tensor(label_list, dtype=torch.float).reshape(1, -1).to(device)
        g = a5[i] - label_list
        size1 = w[5].shape[0]
        size2 = w[5].shape[1]
        # 输出层到隐藏层四
        w_5 = w_5 + g.reshape(1, -1) * ((a[4][i].reshape(-1, 1)).repeat(1, size2)) / batch_size
        for j in range(4):
            # 上层权重举证的维度
            size1 = w[5 - j].shape[0]
            size2 = w[5 - j].shape[1]
            b = [elem * (1 - elem) for elem in a[4 - j][i]]
            b = torch.tensor(b, dtype=torch.float).reshape(-1, 1).to(device)
            g = torch.matmul(w[5 - j], g.reshape(-1, 1)) * b
            if j == 0:
                # 隐藏层四到隐藏层三
                w_4 += (g.reshape(1, -1) * (a[3][i].reshape(-1, 1)).repeat(1, size1) / batch_size)
            elif j == 1:
                # 隐藏层三到隐藏层二
                w_3 += (g.reshape(1, -1) * (a[2][i].reshape(-1, 1)).repeat(1, size1) / batch_size)
            elif j == 2:
                # 隐藏层二到隐藏层一
                w_2 += (g.reshape(1, -1) * (a[1][i].reshape(-1, 1)).repeat(1, size1) / batch_size)
            else:
                # 隐藏层一到输入层
                w_1 += (g.reshape(1, -1) * (data[i].reshape(-1, 1)).repeat(1, size1) / batch_size)
    deta_w = {1: w_1, 2: w_2, 3: w_3, 4: w_4, 5: w_5}
    return deta_w


def optimization(deta_w):
    for i in w.keys():
        w[i] = w[i] - lr * deta_w[i]


def train_epoch():
    epoch_loss = 0
    epoch_accuracy = 0
    for i, batch in zip(tqdm(range(len(trainIter))), trainIter):
        data = batch[0].to(device)
        label = batch[1].to(device)
        # 前向传播
        output, a = forward(data)
        # 计算损失
        loss = CrossEntropy(output, label)
        loss = torch.mean(loss)
        # 计算正确率
        acc = accuracy(output, label)
        # 反向传播
        deta_w = backward(output, data, label, a)
        # 梯度更新
        optimization(deta_w)
        epoch_loss += loss.item()
        epoch_accuracy += acc
    return epoch_loss / len(trainIter), epoch_accuracy / len(trainIter)


def evaluate():
    epoch_loss = 0
    epoch_accuracy = 0
    for batch in testIter:
        data = batch[0].to(device)
        label = batch[1].to(device)
        # 前向传播
        output, _ = forward(data)
        # 计算损失
        loss = CrossEntropy(output, label)
        loss = torch.mean(loss)
        # 计算正确率
        acc = accuracy(output, label)
        epoch_loss += loss.item()
        epoch_accuracy += acc
    return epoch_loss / len(testIter), epoch_accuracy / len(testIter)


def train():
    for epoch in range(epochs):
        start = time.time()
        train_loss, train_accuracy = train_epoch()
        validation_loss, validation_accuracy = evaluate()
        end = time.time()
        print('Epoch:[{}/{}]'.format(epoch + 1, epochs))
        print('Train Loss : {:.3f}  | Train Accuracy : {:.3f}%'.format(train_loss, train_accuracy * 100))
        print('Validation Loss : {:.3f}  | Validation Accuracy : {:.3f}%'.format(validation_loss,
                                                                                 validation_accuracy * 100))
        print('Time: {} s'.format(end - start))


def predict(image, label):
    """
    预测一张图片的标签
    :param image: 输入一张图片（宽，高）
    :return:
    """
    width = image.shape[0]
    hight = image.shape[1]
    data = image.reshape(-1, width * hight)
    output, _ = forward(data)
    predicted = output.argmax(dim=1)
    print('模型预测该图片的数字是：{}'.format(predicted[0]))
    print('真实的标签是：{}'.format(label.item()))


if __name__ == '__main__':
    path = 'D:\\wodedaima\\python\\course_works\\DeepLearningExperiment\\data'
    trainIter, testIter = DataSet(path)
    device = torch.device('cuda:0')
    # 各层中神经元的古树
    input_size = 28 * 28
    h1 = 512
    h2 = 256
    h3 = 128
    h4 = 64
    output_size = 10
    flag = True
    if flag:
        w = np.load('model1.npy', allow_pickle=True).item()
    else:
        # 初始化权重
        w1 = torch.randn(input_size, h1).to(device)
        w2 = torch.randn(h1, h2).to(device)
        w3 = torch.randn(h2, h3).to(device)
        w4 = torch.randn(h3, h4).to(device)
        w5 = torch.randn(h4, output_size).to(device)
        w = {1: w1, 2: w2, 3: w3, 4: w4, 5: w5}
    # 参数
    lr = 1.5
    epochs = 20
    is_train = False
    if is_train:
        # --------------训练-------------
        print('---------------训练---------------')
        train()
        np.save('model1.npy', w)
    else:
        # -------------预测------------------
        #path = "C:\\Users\\86158\\Desktop\\1.jpg"
        #image = cv.imread(path)
        #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #image = cv.resize(image, (28, 28))
        #cv.imshow('9', image)
        #cv.waitKey(10000)
        #cv.destroyAllWindows()
        #image = torch.tensor(image, dtype=torch.float).to(device)
        #label = torch.LongTensor([5])
        test_data = next(iter(testIter))
        image = test_data[0][0].squeeze().to(device)
        f = transforms.ToPILImage()
        a = f(image)
        a.show()
        label = test_data[1][0].to(device)
        predict(image, label)
