import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.switch_backend('TkAgg')


def createData():
    # 加载数据
    df = pd.read_csv('./data.csv')
    # 提取数据
    values = df.values
    name = df.columns[:-1]
    data = values[:, :-1]
    label = values[:, -1]
    # 转换为numpy类型
    data = np.asarray(data)
    name = np.asarray(name).reshape(1, -1)
    label = np.asarray(label).reshape(1, -1)
    return data, label, name


def splitData(Data, Label):
    Label = Label.reshape(-1, 1)
    DataTrain, DataTest, LabelTrain, LabelTest = train_test_split(Data, Label, test_size=0.3)
    return DataTrain, DataTest, LabelTrain, LabelTest


# 定义一个常用函数 用来求numpy array中数值等于某值的元素数量
equalNums = lambda x, y: 0 if x is None else x[x == y].size


def entropy(label):
    """
    计算一个集合中所有样本的熵
    :param label:输入样本集合的标签
    :return:熵
    """
    _, counts = np.unique(label, return_counts=True)
    probabilities = counts / len(label)
    Entropy = -np.sum(probabilities * np.log2(probabilities))
    return Entropy


def Gini(label):
    # 转化为ndarray
    label = np.asarray(label)
    _, counts = np.unique(label, return_counts=True)
    probabilities = counts / len(label)
    Gini = np.sum(probabilities ** 2)
    return 1 - Gini


def information_gain(label, left_label, right_label):
    """
    计算信息增益
    :param label: 划分前节点的样本
    :param left_label: 划分后左孩子节点的样本
    :param right_label: 划分后右孩子节点的样本
    :return: 信息增益
    """
    # 计算双亲节点的信息熵
    parent_entropy = entropy(label)
    left_weight = len(left_label) / len(label)
    right_weight = len(right_label) / len(label)
    children_entropy = (left_weight * entropy(left_label) +
                        right_weight * entropy(right_label))
    return parent_entropy - children_entropy


def find_best_split(data, label, method):
    """
    当特征值为离散取值时，采用二分法，对双亲节点的样本进行划分，找到使得信息增益最大的划分值
    :param method: 使用什么方法对节点进行划分
    :param data:划分前样本的取值
    :param label:划分前样本的标签
    :return:划分后使信息增益最大的特征和划分阈值
    """
    m, n = data.shape  # m为样本个数，n为特征个数
    best_feature = None
    best_threshold = None
    # 使用信息熵
    if method == 'Entropy':
        best_info_gain = 0
    # 使用基尼指数
    elif method == 'Gini':
        best_Gini_Index = np.Inf
    # 对当前样本的所有特征进行遍历
    for feature in range(n):
        # 取出特征为feature的所有样本值
        X_feature = data[:, feature]
        unique_X = np.unique(X_feature)
        # 找到最佳的二分点
        thresholds = []
        for i in range(len(unique_X) - 1):
            thresholds.append((unique_X[i] + unique_X[i + 1]) / 2)
        for threshold in thresholds:
            # 小于阈值的是左分支的样本
            left_mask = X_feature <= threshold
            # 大于分支的是右分支的样本
            right_mask = X_feature > threshold
            if len(label[left_mask]) == 0 or len(label[right_mask]) == 0:
                continue
            if method == 'Entropy':
                info_gain = information_gain(label, label[left_mask], label[right_mask])
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

            elif method == 'Gini':
                w1 = len(label[left_mask]) / len(label)
                w2 = len(label[right_mask]) / len(label)
                Gini_Index = w1 * Gini(label[left_mask]) + w2 * Gini(label[right_mask])
                if Gini_Index < best_Gini_Index:
                    best_Gini_Index = Gini_Index
                    best_feature = feature
                    best_threshold = threshold
    return best_feature, best_threshold


# 多数投票
def voteLabel(labels):
    uniqLabels = list(set(labels))
    labels = np.asarray(labels)

    finalLabel = 0
    labelNum = []
    for label in uniqLabels:
        # 统计每个标签值得数量
        labelNum.append(equalNums(labels, label))
    # 返回数量最大的标签
    return uniqLabels[labelNum.index(max(labelNum))]


# 创建决策树
def createTree(data, labels, names, method='Entropy'):
    # 每次递归赋值一份数据
    data = np.asarray(data)
    labels = np.asarray(labels)
    names = np.asarray(names)
    # 如果样本data中全属于同一类别
    if len(set(labels)) == 1:
        return labels[0]
    # 如果没有划分的属性
    elif data.size == 0:
        return voteLabel(labels)
    # 其他情况则选取特征
    bestFeat, threshold = find_best_split(data, labels, method=method)
    # 取特征名称
    bestFeatName = names[0, bestFeat]
    # 根据选取的特征名称创建树节点
    decisionTree = {bestFeatName: {}}
    # 划分数据集左右
    left_mask = data[:, bestFeat] <= threshold
    right_mask = data[:, bestFeat] > threshold
    # 对最优特征的每个特征值所分的左右数据子集进行计算
    # 左树
    featValue_left = '<={:.2f}'.format(threshold.item())
    decisionTree[bestFeatName][featValue_left] = createTree(data[left_mask], labels[left_mask], names,
                                                            method)
    # 右树
    featValue_right = '   >{:.2f}'.format(threshold.item())
    decisionTree[bestFeatName][featValue_right] = createTree(data[right_mask], labels[right_mask], names,
                                                             method)
    return decisionTree


# 树信息统计 叶子节点数量 和 树深度
def getTreeSize(decisionTree):
    nodeName = list(decisionTree.keys())[0]
    nodeValue = decisionTree[nodeName]
    leafNum = 0
    treeDepth = 0
    leafDepth = 0
    for val in nodeValue.keys():
        if type(nodeValue[val]) == dict:
            leafNum += getTreeSize(nodeValue[val])[0]
            leafDepth = 1 + getTreeSize(nodeValue[val])[1]
        else:
            leafNum += 1
            leafDepth = 1
        treeDepth = max(treeDepth, leafDepth)
    return leafNum, treeDepth


decisionNodeStyle = dict(boxstyle="square", fc="0.8")
leafNodeStyle = {"boxstyle": "round4", "fc": "0.8"}
arrowArgs = {"arrowstyle": "<-"}


# 画节点
def plotNode(nodeText, centerPt, parentPt, nodeStyle):
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords="axes fraction", xytext=centerPt
                            , textcoords="axes fraction", va="center", ha="center", bbox=nodeStyle,
                            arrowprops=arrowArgs)


# 添加箭头上的标注文字
def plotMidText(centerPt, parentPt, lineText):
    xMid = (centerPt[0] + parentPt[0]) / 2.0
    yMid = (centerPt[1] + parentPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, lineText)


# 画树
def plotTree(decisionTree, parentPt, parentValue):
    # 计算宽与高
    leafNum, treeDepth = getTreeSize(decisionTree)
    # 在 1 * 1 的范围内画图，因此分母为 1
    # 每个叶节点之间的偏移量
    plotTree.xOff = plotTree.figSize / (plotTree.totalLeaf - 1)
    # 每一层的高度偏移量
    plotTree.yOff = plotTree.figSize / plotTree.totalDepth
    # 节点名称
    nodeName = list(decisionTree.keys())[0]
    # 根节点的起止点相同，可避免画线；如果是中间节点，则从当前叶节点的位置开始，
    #      然后加上本次子树的宽度的一半，则为决策节点的横向位置
    centerPt = (plotTree.x + (leafNum - 1) * plotTree.xOff / 2.0, plotTree.y)
    # 画出该决策节点
    plotNode(nodeName, centerPt, parentPt, decisionNodeStyle)
    # 标记本节点对应父节点的属性值
    plotMidText(centerPt, parentPt, parentValue)
    # 取本节点的属性值
    treeValue = decisionTree[nodeName]
    # 下一层各节点的高度
    plotTree.y = plotTree.y - plotTree.yOff
    # 绘制下一层
    for val in treeValue.keys():
        # 如果属性值对应的是字典，说明是子树，进行递归调用； 否则则为叶子节点
        if type(treeValue[val]) == dict:
            plotTree(treeValue[val], centerPt, str(val))
        else:
            plotNode(treeValue[val], (plotTree.x, plotTree.y), centerPt, leafNodeStyle)
            plotMidText((plotTree.x, plotTree.y), centerPt, str(val))
            # 移到下一个叶子节点
            plotTree.x = plotTree.x + plotTree.xOff
    # 递归完成后返回上一层
    plotTree.y = plotTree.y + plotTree.yOff


# 画出决策树
def createPlot(decisionTree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = {"xticks": [], "yticks": []}
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 定义画图的图形尺寸
    plotTree.figSize = 1
    # 初始化树的总大小
    plotTree.totalLeaf, plotTree.totalDepth = getTreeSize(decisionTree)
    # 叶子节点的初始位置x 和 根节点的初始层高度y
    plotTree.x = 0
    plotTree.y = plotTree.figSize
    plotTree(decisionTree, (plotTree.figSize / 2.0, plotTree.y), "")
    plt.show()


# 创建预剪枝决策树
def createTreePrePruning(dataTrain, labelTrain, dataTest, labelTest, names, method='Entropy'):
    trainData = np.asarray(dataTrain)
    labelTrain = np.asarray(labelTrain)
    testData = np.asarray(dataTest)
    labelTest = np.asarray(labelTest)
    names = np.asarray(names)
    # 如果结果为单一结果
    if len(set(labelTrain)) == 1:
        return labelTrain[0]
        # 如果没有待分类特征
    elif trainData.size == 0:
        return voteLabel(labelTrain)
    # 其他情况则选取特征
    bestFeat, threshold = find_best_split(dataTrain, labelTrain, method=method)
    # 取特征名称
    bestFeatName = names[0, bestFeat]
    # 预剪枝评估
    # 划分前的分类标签
    labelTrainLabelPre = voteLabel(labelTrain)
    # 划分前的测试标签正确比例
    labelTestRatioPre = equalNums(labelTest, labelTrainLabelPre) / labelTest.size
    # 划分后 每个特征值的分类标签正确的数量
    labelTrainEqNumPost = 0
    left_mask_test = dataTest[:, bestFeat] <= threshold
    right_mask_test = dataTest[:, bestFeat] > threshold
    dataLabel_left = labelTest[left_mask_test]
    dataLabel_right = labelTest[right_mask_test]
    if len(dataLabel_left) == 0:
        pass
    else:
        labelTrainEqNumPost += equalNums(dataLabel_left, voteLabel(dataLabel_left)) + 0.0
    if len(dataLabel_right) == 0:
        pass
    else:
        labelTrainEqNumPost += equalNums(dataLabel_right, voteLabel(dataLabel_right)) + 0.0
    # 划分后 正确的比例
    labelTestRatioPost = labelTrainEqNumPost / labelTest.size
    # 如果划分后的精度相比划分前的精度下降, 则直接作为叶子节点返回
    if labelTestRatioPost <= labelTestRatioPre:
        return labelTrainLabelPre
    else:
        decisionTree = {bestFeatName: {}}
        # 划分数据集左右
        left_mask = dataTrain[:, bestFeat] <= threshold
        right_mask = dataTrain[:, bestFeat] > threshold
        # 左树
        featValue_left = '<={:.2f}'.format(threshold.item())
        decisionTree[bestFeatName][featValue_left] = createTreePrePruning(dataTrain[left_mask],
                                                                          labelTrain[left_mask],
                                                                          dataTest[left_mask_test],
                                                                          labelTest[left_mask_test],
                                                                          names,
                                                                          method)
        # 右树
        featValue_right = '   >{:.2f}'.format(threshold.item())
        decisionTree[bestFeatName][featValue_right] = createTreePrePruning(dataTrain[right_mask],
                                                                           labelTrain[right_mask],
                                                                           dataTest[right_mask_test],
                                                                           labelTest[right_mask_test],
                                                                           names,
                                                                           method)
    return decisionTree


# 创建决策树 带预划分标签
def createTreeWithLabel(data, labels, names, method='Gini'):
    data = np.asarray(data)
    labels = np.asarray(labels)
    names = np.asarray(names)
    # 如果不划分的标签为
    votedLabel = voteLabel(labels)
    # 如果结果为单一结果
    if len(set(labels)) == 1:
        return votedLabel
        # 如果没有待分类特征
    elif data.size == 0:
        return votedLabel
    # 其他情况则选取特征
    bestFeat, threshold = find_best_split(data, labels, method=method)
    # 取特征名称
    bestFeatName = names[0, bestFeat]
    # 根据选取的特征名称创建树节点 划分前的标签votedPreDivisionLabel=_vpdl
    decisionTree = {bestFeatName: {}}
    decisionTree[bestFeatName]["_vpdl"] = votedLabel
    # 划分数据集左右
    left_mask = data[:, bestFeat] <= threshold
    right_mask = data[:, bestFeat] > threshold
    # 对最优特征的每个特征值所分的左右数据子集进行计算
    # 左树
    featValue_left = '<={:.2f}'.format(threshold.item())
    decisionTree[bestFeatName][featValue_left] = createTreeWithLabel(data[left_mask], labels[left_mask], names,
                                                                     method)
    # 右树
    featValue_right = '   >{:.2f}'.format(threshold.item())
    decisionTree[bestFeatName][featValue_right] = createTreeWithLabel(data[right_mask], labels[right_mask], names,
                                                                      method)
    return decisionTree


# 将带预划分标签的tree转化为常规的tree
def convertTree(labeledTree):
    labeledTreeNew = labeledTree.copy()
    # 结点名称
    nodeName = list(labeledTree.keys())[0]
    # 根节点的子树
    labeledTreeNew[nodeName] = labeledTree[nodeName].copy()
    for val in list(labeledTree[nodeName].keys()):
        # 预先标记的标签
        if val == "_vpdl":
            labeledTreeNew[nodeName].pop(val)
        elif type(labeledTree[nodeName][val]) == dict:
            labeledTreeNew[nodeName][val] = convertTree(labeledTree[nodeName][val])
    return labeledTreeNew


def extract_numbers_from_string(string):
    numbers = ""
    for char in string:
        if char == '<':
            pass
        elif char == '>':
            pass
        elif char == '=':
            pass
        else:
            numbers += char
    return float(numbers)


# 后剪枝 训练完成后决策节点进行替换评估
def treePostPruning(labeledTree, dataTest, labelTest, names):
    newTree = labeledTree.copy()
    dataTest = np.asarray(dataTest)
    labelTest = np.asarray(labelTest)
    names = np.asarray(names)
    # 取决策节点的名称 即特征的名称
    featName = list(labeledTree.keys())[0]
    featCol = np.where(names[0] == featName)[0][0]
    # 获取子树
    newTree[featName] = labeledTree[featName].copy()
    featValueDict = newTree[featName]
    # 根节点的预先标签
    featPreLabel = featValueDict.pop('_vpdl')
    subTreeFlag = 0
    # 分割测试数据 如果有数据 则进行测试或递归调用
    dataFlag = 1 if sum(dataTest.shape) > 0 else 0
    # 区分左右子树
    leftFlag = 1
    for featValue in featValueDict.keys():
        threshold = extract_numbers_from_string(featValue)
        # 划分数据集左右
        left_mask = dataTest[:, featCol] <= threshold
        right_mask = dataTest[:, featCol] > threshold
        if dataFlag == 1 and type(featValueDict[featValue]) == dict:
            subTreeFlag = 1
            # 如果是子树则递归
            if leftFlag == 1:
                newTree[featName][featValue] = treePostPruning(featValueDict[featValue], dataTest[left_mask],
                                                               labelTest[left_mask], names)
                leftFlag = 0
            else:
                newTree[featName][featValue] = treePostPruning(featValueDict[featValue], dataTest[right_mask],
                                                               labelTest[right_mask], names)
            # 如果递归后为叶子 则后续进行评估
            if type(featValueDict[featValue]) != dict:
                subTreeFlag = 0

        # 如果没有数据  则转换子树
        if dataFlag == 0 and type(featValueDict[featValue]) == dict:
            subTreeFlag = 1
            newTree[featName][featValue] = convertTree(featValueDict[featValue])
    # 如果没有子树
    if subTreeFlag == 0 and labelTest.size != 0:
        # 后剪枝后根节点的正确率
        ratioPreDivision = equalNums(labelTest, featPreLabel) / labelTest.size
        # 剪枝前根结点的正确率
        equalNum = 0
        leftFlag = 1
        for val in featValueDict.keys():
            threshold = extract_numbers_from_string(val)
            # 划分数据集左右
            left_mask = dataTest[:, featCol] <= threshold
            right_mask = dataTest[:, featCol] > threshold
            if leftFlag == 1:
                equalNum += equalNums(labelTest[left_mask], featValueDict[val])
                leftFlag = 0
            else:
                equalNum += equalNums(labelTest[right_mask], featValueDict[val])
        ratioAfterDivision = equalNum / labelTest.size
        if ratioAfterDivision < ratioPreDivision:
            newTree = featPreLabel
    return newTree


if __name__ == '__main__':
    Data, Label, Name = createData()
    dataTrain, dataTest, labelTrain, labelTest = splitData(Data, Label)
    # 修改label形状
    labelTrain = labelTrain.reshape(-1)
    labelTest = labelTest.reshape(-1)
    # 信息熵请使用‘Entropy’, 基尼指数请使用‘Gini’
    Tree_Entropy = createTree(dataTrain, labelTrain, Name, method='Entropy')
    print(Tree_Entropy)
    createPlot(Tree_Entropy)
    # Gini指数
    Tree_Gini = createTree(dataTrain, labelTrain, Name, method='Gini')
    print(Tree_Gini)
    createPlot(Tree_Gini)
    # 预剪枝
    TreePrePruning = createTreePrePruning(dataTrain, labelTrain, dataTest, labelTest, Name, method='Gini')
    print(TreePrePruning)
    createPlot(TreePrePruning)
    # 后剪枝
    TreeWithLabel = createTreeWithLabel(dataTrain, labelTrain, Name)
    TreePostPruning = treePostPruning(TreeWithLabel, dataTest, labelTest, Name)
    print(TreePostPruning)
    createPlot(TreePostPruning)
