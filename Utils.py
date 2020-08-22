import numpy as np


class Utils(object):
    @staticmethod
    def sigmoid(array):
        return 1 / (1 + np.exp(array))

    @staticmethod
    def sigmoid_derivative(array):
        sigmoid_value = 1 / (1 + np.exp(array))
        return np.multiply(sigmoid_value, 1 - sigmoid_value)

def softmax(x):
    """
    对输入x的每一行计算softmax。
    该函数对于输入是向量（将向量视为单独的行）或者矩阵（M x N）均适用。
    代码利用softmax函数的性质: softmax(x) = softmax(x + c)
    参数:
    x -- 一个N维向量，或者M x N维numpy矩阵.
    返回值:
    x -- 在函数内部处理后的x
    """
    orig_shape = x.shape

    # 根据输入类型是矩阵还是向量分别计算softmax
    if len(x.shape) > 1:
        # 矩阵
        tmp = np.max(x, axis=1)  # 得到每行的最大值，用于缩放每行的元素，避免溢出
        x -= tmp.reshape((x.shape[0], 1))  # 利用性质缩放元素
        x = np.exp(x)  # 计算所有值的指数
        tmp = np.sum(x, axis=1)  # 每行求和
        x /= tmp.reshape((x.shape[0], 1))  # 求softmax
    else:
        # 向量
        tmp = np.max(x)  # 得到最大值
        x -= tmp  # 利用最大值缩放数据
        x = np.exp(x)  # 对所有元素求指数
        tmp = np.sum(x)  # 求元素和
        x /= tmp  # 求somftmax
    return x

# def softmax(x):
#     max = np.max(x)
#     return np.exp(x - max) / sum(np.exp(x - max))


def der_softmax_cross_entropy(act_array, pre_array):
    # y_act = onehot(num_class,label_array)
    y_act = act_array
    y_hat = pre_array
    return y_hat - y_act


def sigmoid(x):
    """sigmoid函数"""
    if x.any() >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def der_sigmoid(x):
    """sigmoid函数的导数"""
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    """tanh函数"""
    return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))


def der_tanh(x):
    """tanh函数的导数"""
    return 1 - tanh(x) * tanh(x)


def relu(x):
    """relu函数"""
    temp = np.zeros_like(x)
    if_bigger_zero = (x > temp)
    return x * if_bigger_zero


def der_relu(x):
    """relu函数的导数"""
    temp = np.zeros_like(x)
    if_bigger_equal_zero = (x >= temp)
    return if_bigger_equal_zero * np.ones_like(x)


def onehot(num_class, label_array):
    return np.eye(num_class)[label_array]
