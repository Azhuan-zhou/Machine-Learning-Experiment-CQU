import numpy as np
from PCA import read_from_csv, PCA_method
import matplotlib.pyplot as plt


class cluster:
    def __init__(self, featNum, x):
        """
        初始化一个cluster
        :param x: 簇初始化的样本
        """
        self.num_members = 0  # 簇中样本个数
        self.means = x[0:4]  # 一个簇的均值向量
        self.members = []  # cluster中的成员
        self.featNum = featNum  # 特征个数
        self.species = {}  # cluster中的类别

    def add(self, member):
        # 离该cluster最近的样本加入进来 （feat1,feat2,feat3,feat4,species）
        self.members.append(member)
        self.num_members = self.num_members + 1  # 更新cluster中样本的个数

    def new(self):
        """
        更新簇的均值向量
        :return:
        """
        data = self.get_data_numpy()
        new_means = np.average(data, axis=0)
        flag = (self.means == new_means).sum()
        # 当簇的均值向量不更新的时候，返回1
        if flag == self.featNum:
            return 1
        else:
            self.means = new_means
            return 0

    def distance(self, x):
        # 计算某一个样本x(ndarray)（feat1,feat2,feat3,feat4,species）和cluster的距离
        distance = np.sum(np.square(self.means - x[0:4]))
        return distance

    def clear(self):
        # 重置cluster的成员
        self.members = []
        self.num_members = 0

    def get_accuracy(self):
        samples = self.get_spices_numpy()
        species = set(samples)
        count = 0
        max_count = 0
        for specie in species:
            count = (samples == specie).sum()
            self.species[specie] = count
        max_count = max(self.species.values())
        max_specie = [key for key, value in self.species.items() if value == max_count]
        species_num = len(samples)
        return max_specie, max_count / species_num * 100

    def get_data_numpy(self):
        """
        从member（feat1,feat2,feat3,feat4,species）中提取出数据的部分
        :return:
        """
        return np.asarray(self.members)[:, 0:4].astype('float32')

    def get_spices_numpy(self):
        """
        从member（feat1,feat2,feat3,feat4,species）中提取出类别的部分
        :return:
        """
        return np.array(self.members)[:, 4]


def KMeans(data, featNum, p):
    """
    K均值算法
    :param featNum: 数据的维度
    :param data: 数据 (num_samples,5)（feat1,feat2,feat3,feat4,species）
    :param p: 聚类时簇的个数
    :return: clusters
    """
    data_num = len(data)
    clusters = [cluster(featNum, data[i]) for i in range(p)]  # 产生p个cluster
    flag = 0
    times = 0  # 记录迭代次数
    while flag != p:
        flag = 0
        # 先重置所有的clusters
        for k in range(p):
            clusters[k].clear()
        for i in range(data_num):
            # 计算样本i与各个簇的距离
            dist = np.Inf
            index = 0  # 记录距离最短的簇
            for j, cluster_ in enumerate(clusters):
                cur_dist = cluster_.distance(data[i])
                if cur_dist < dist:  # 寻找最小距离
                    dist = cur_dist
                    index = j
            clusters[index].add(data[i])
        # 对所有的cluster更新均值向量
        for k in range(p):
            flag += clusters[k].new()
        times += 1

    return clusters, times


def main(path, p):
    data, labels, class_names = read_from_csv(path)
    featNum = data.shape[-1]  # 数据维度
    dataWithLabels = np.concatenate((data, labels.reshape(-1, 1)), axis=1)
    clusters, times = KMeans(dataWithLabels, featNum, p)
    print("K-means一共迭代：{}次".format(times))
    # 计算主成分
    _, w1, w2 = PCA_method(data)
    w = np.concatenate((w1, w2), axis=1)  # 降维矩阵
    plt.figure(figsize=(6, 6))
    plt.title('K-means, times:{}'.format(times))
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.grid(True)
    color = ['green', 'blue', 'red']
    a1, a2, a3 = None, None, None
    a = [a1, a2, a3]
    for i, cluster_ in enumerate(clusters):
        specie, accuracy = cluster_.get_accuracy()
        print("Cluster{}中, 一共有{}个类别，分别是{}，出现最多的鸢尾花类别是 {}, 聚类的正确率为：{}%".
              format(i + 1, len(cluster_.species), cluster_.species, specie, accuracy))
        center = np.matmul(cluster_.means, w)
        plt.scatter(center[0], center[1], c='black', marker='+', s=80)
        dataInCluster = np.asarray(cluster_.members)[:, 0:4].astype('float32')
        reductionDatas = np.matmul(dataInCluster, w)
        for j, reductionData in enumerate(reductionDatas):
            a[i] = plt.scatter(reductionData[0], reductionData[1], c=color[i], s=20)
    plt.legend((a[0], a[1], a[2]), ('Cluster{}'.format(1), 'Cluster{}'.format(2), 'Cluster{}'.format(3)), loc='best')
    plt.show()


if __name__ == '__main__':
    main('./iris.csv', 3)
