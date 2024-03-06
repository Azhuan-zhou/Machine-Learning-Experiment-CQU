import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

plt.switch_backend('TkAgg')


def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪声"""
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.8, y.shape)
    return X, y.reshape((-1, 1))


def plot_dots(feature, label, w):
    x = feature[:, 0]
    y = feature[:, 1]
    z = label
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    ax.scatter(x, y, z, c='red', label='dots')
    plt.legend()
    # 定义平面上的点
    x_plane = np.linspace(-2.5, 2.5, 100)
    y_plane = np.linspace(-2.5, 2.5, 100)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    a = w[0, 0]
    b = w[1, 0]
    c = w[2, 0]
    Z_plane = a * X_plane + b * Y_plane + c
    ax.plot_surface(X_plane, Y_plane, Z_plane, color='royalblue')
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()


def plotdots_2d(x, y, w, b):
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c='red', s=2, label='dots')
    plt.legend()
    x_hat = np.linspace(-3, 3, 50)
    y_hat = w * x_hat + b
    str = 'y = {}x+{}'.format(w, b)
    plt.plot(x_hat, y_hat, color='blue',label=str)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def calculate_parameter_2(feature, label):
    startTime = time.time()
    x = [np.append(f, 1) for f in feature]
    x = np.array(x)
    x_x = np.matrix(x.T @ x)
    x_x_inverse = np.linalg.inv(x_x)
    w = x_x_inverse @ x.T @ label
    endTime = time.time()
    return w, endTime - startTime


def calculate_parameter_1(x, y):
    startTime = time.time()
    l = len(x)
    x_y_hat = (x * y).sum() / l
    x_hat_y_hat = (x.sum() / l) * (y.sum() / l)
    x_square_hat = (x ** 2).sum() / l
    x_hat_square = (x.sum() / l) ** 2
    w = (x_y_hat - x_hat_y_hat) / (x_square_hat - x_hat_square)
    b = y.sum() / l - w * (x.sum() / l)
    endTime = time.time()
    return w, b, endTime - startTime


if __name__ == '__main__':
    # 是一个一维输入
    true_w_1 = np.array([4])
    true_b_1 = 4.2
    features, labels = synthetic_data(true_w_1, true_b_1, 1000)
    print('True w is {}, True b is {}'.format(true_w_1, true_b_1))
    w_1, b_1, time_1 = calculate_parameter_1(features, labels)
    print('Predicted w is {}, predicted b is {} and the method one costs {} seconds'.format(w_1, b_1, time_1))
    plotdots_2d(features, labels, w_1, b_1)
    # 是一个二维的输入
    true_w_2 = np.array([4, 4.9])
    true_b_2 = 4.2
    features, labels = synthetic_data(true_w_2, true_b_2, 1000)
    print('True w is {}, True b is {}'.format(true_w_2, true_b_2))
    w_2, time_2 = calculate_parameter_2(features, labels)
    print(
        'Predicted w is {}, predicted b is {} and the method two costs {} seconds'.format([w_2[0][0, 0], w_2[1][0, 0]],
                                                                                          w_2[2][0, 0], time_2))
    plot_dots(features, labels, w_2)
