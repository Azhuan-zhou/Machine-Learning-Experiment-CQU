import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    data = df.values
    features = df.columns[:-1]
    x = data[:, :-1]
    y = data[:, -1]
    return x, y, list(features)


def maxY(Y):
    y, counts = np.unique(Y, return_counts=True)
    index = np.argemax(counts)
    return y[index]


def splitDataset(X, best_feature):
    retX = []
    for x in X:  # 遍历数据集
        reducedVec = np.concatenate((x[:best_feature], x[best_feature + 1:]), axis=0)
        retX.append(reducedVec)
    return np.array(retX)


def splitFeatures(Features, bestFeature):
    Features = np.delete(Features, bestFeature)
    return Features


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        初始化决策树
        :param max_depth: 树的最大深度
        :param min_samples_split: 节点产生分支的最小样本个数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def entropy(self, y):
        """
        计算一个集合中所有样本的熵
        :param y:输入样本集合的标签
        :return:熵
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def information_gain(self, y_labels, left_y, right_y):
        """
        计算信息增益
        :param y_labels: 划分前节点的样本
        :param left_y: 划分后左孩子节点的样本
        :param right_y: 划分后右孩子节点的样本
        :return: 信息增益
        """
        # 计算双亲节点的信息熵
        parent_entropy = self.entropy(y_labels)
        left_weight = len(left_y) / len(y_labels)
        right_weight = len(right_y) / len(y_labels)
        children_entropy = (left_weight * self.entropy(left_y) +
                            right_weight * self.entropy(right_y))
        return parent_entropy - children_entropy

    def find_best_split(self, X_samples, y_labels):
        """
        当特征值为离散取值时，采用二分法，对双亲节点的样本进行划分，找到使得信息增益最大的划分值
        :param X_samples:划分前样本的取值
        :param y_labels:划分前样本的标签
        :return:划分后使信息增益最大的特征和划分阈值
        """
        m, n = X_samples.shape  # m为样本个数，n为特征个数
        parent_entropy = self.entropy(y_labels)
        best_info_gain = 0
        best_feature = None
        best_threshold = None
        # 对当前样本的所有特征进行遍历
        for feature in range(n):
            # 取出特征为feature的所有样本值
            X_feature = X_samples[:, feature]
            unique_X = np.unique(X_feature)
            thresholds = []
            for i in range(len(unique_X) - 1):
                thresholds.append((unique_X[i] + unique_X[i + 1]) / 2)
            for threshold in thresholds:
                # 小于阈值的是左分支的样本
                left_mask = X_feature <= threshold
                # 大于分支的是右分支的样本
                right_mask = X_feature > threshold
                if len(y_labels[left_mask]) == 0 or len(y_labels[right_mask]) == 0:
                    continue
                info_gain = self.information_gain(y_labels, y_labels[left_mask], y_labels[right_mask])
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def fit_process(self, X_samples, y_labels, Features, depth=0):
        # 如果样本X中中全属于同一类别
        if np.sum(y_labels == y_labels[0]) == len(y_labels):
            return y_labels[0]
        # 如果没有划分的属性或者在所有样本中的所有属性的取值都相同
        if len(X_samples[0]) == 1:
            return maxY(y_labels)
        best_feature, best_threshold = self.find_best_split(X_samples, y_labels)

        if best_feature is not None:
            left_mask = X_samples[:, best_feature] <= best_threshold
            right_mask = X_samples[:, best_feature] > best_threshold
            if len(y_labels[left_mask]) > 0 and len(y_labels[right_mask]) > 0:
                tree = {'feature_index': Features[best_feature], 'threshold': best_threshold,
                        'left': self.fit_process(splitDataset(np.copy(X_samples[left_mask]),
                                                              best_feature,
                                                              ),
                                                 y_labels[left_mask],
                                                 np.copy(splitFeatures(Features, best_feature)),
                                                 depth + 1),
                        'right': self.fit_process(splitDataset(np.copy(X_samples[right_mask]),
                                                               best_feature,
                                                               ),
                                                  y_labels[right_mask],
                                                  np.copy(splitFeatures(Features, best_feature)),
                                                  depth + 1)}
        return tree

    def fit(self, X_samples, y_labels, Features, depth=0):
        self.tree = self.fit_process(X_samples, y_labels, Features, depth)

    def predict_one(self, x, features,tree=None,):
        if tree is None:
            tree = self.tree
        if isinstance(tree, float):
            return tree
        if 'threshold' in tree:
            if x[features.index(tree['feature_index'])] <= tree['threshold']:
                return self.predict_one(x, features, tree['left'])
            else:
                return self.predict_one(x, tree['right'])
    def predict(self, X_samples, features):
        predictions = [self.predict_one(x, features) for x in X_samples]
        return predictions


path = './data.csv'
X, y, features = load_data(path)
# 创建并训练决策树模型
tree_classifier = DecisionTree()
tree_classifier.fit(X, y, features)

# 进行预测
predictions = tree_classifier.predict(
    np.array([[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]]),
    list(features)
)
print(predictions)  # 输出：[0, 1, 1, 0]
