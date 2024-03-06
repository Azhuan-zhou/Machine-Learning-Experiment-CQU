import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('TKAgg')


def read_from_csv(path):
    dataframe = pd.read_csv(path)
    data = dataframe.values[:, 0:4]
    labels = dataframe.values[:, 4]
    classes_names = dataframe.columns[0:4]
    return data.astype('float32'), np.array(labels), np.array(classes_names)


def PCA_method(data):
    # data (n,p)
    data_avg = np.average(data, axis=0)  # 计算每个特征的均值
    # print(data_avg.shape)
    data = data - data_avg
    data_cov = np.matmul(data.T, data) / (len(data) - 1)  # 计算样本方差
    eigValues, eig = np.linalg.eig(data_cov)
    eigValues_sum = np.sum(eigValues)
    lambda_ = (eigValues[0] + eigValues[1]) / eigValues_sum * 100  # 计算累计贡献率
    w1 = eig[:, 0]
    w2 = eig[:, 1]
    # print('选取的第一个特征值为：{}，特征向量为：{}'.format(eigValues[0], w1))
    # print('选取的第二个特征值为：{}，特征向量为：{}'.format(eigValues[1], w2))
    return lambda_, w1.reshape(-1, 1), w2.reshape(-1, 1)


def main(path):
    plt.figure(figsize=(6, 6))
    plt.title('Iris')
    plt.xlabel('w1')
    plt.ylabel('w2')
    data, labels, classes_names = read_from_csv(path)
    label = list(set(labels))
    # 计算主成分
    lambda_, w1, w2 = PCA_method(data)
    print('累计贡献率为： {}%'.format(lambda_))
    w = np.concatenate((w1, w2), axis=1)
    new_data = np.matmul(data, w)
    color = ['red', 'blue', 'green']
    for i in range(len(new_data)):
        if labels[i] == label[0]:
            a1 = plt.scatter(new_data[i][0], new_data[i][1], c=color[0])
        elif labels[i] == label[1]:
            a2 = plt.scatter(new_data[i][0], new_data[i][1], c=color[1])
        elif labels[i] == label[2]:
            a3 = plt.scatter(new_data[i][0], new_data[i][1], c=color[2])
    plt.legend((a1, a2, a3), (label[0], label[1], label[2]), loc='best')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main('./iris.csv')
