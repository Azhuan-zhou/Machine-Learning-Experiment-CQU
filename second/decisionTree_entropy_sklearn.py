from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd

plt.switch_backend('TkAgg')


def decision_tree(method, x_train, y_train, x_test, y_test, feature_names, target_names, random_state=None,
                  max_depth=None):
    clf = DecisionTreeClassifier(criterion=method,
                                 random_state=random_state,
                                 max_depth=max_depth)
    clf = clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    fig = plt.figure(figsize=(10, 20))
    tree.plot_tree(
        clf,
        feature_names=feature_names,
        class_names=list(target_names),
        filled=True,
        rounded=True
    )
    plt.savefig(method + '.png')
    return accuracy


if __name__ == '__main__':
    wine = datasets.load_wine()
    # 特征名
    features = wine.feature_names
    # 标签
    targets = wine.target_names
    # 将数据写入csv
    flag = False
    if flag:
        df = pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
        outPutPath = './data.csv'
        df.to_csv(outPutPath, index=True, header=features.append('classes'))
    x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)
    # 使用entropy方法
    method_1 = 'entropy'
    a_1 = decision_tree(method_1, x_train, y_train, x_test, y_test, features, targets)
    print('使用Entropy方法的正确率:{}'.format(a_1))
    # 使用gini方法
    method_2 = 'gini'
    a_2 = decision_tree(method_2, x_train, y_train, x_test, y_test, features, targets)
    print('使用Gini方法的正确率:{}'.format(a_2))
    # 预剪枝
    a_3 = decision_tree(method_2, x_train, y_train, x_test, y_test, features, targets,
                        random_state=0,
                        max_depth=4,
                     )
    print('使用Gini方法的正确率:{}'.format(a_3))

