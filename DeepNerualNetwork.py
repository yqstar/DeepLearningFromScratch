# -*- coding:utf-8 -*-
# Author:AndrewYq
# Date:2020-08-16
# Email:hfyqstar@163.com
# Reference: https://mlfromscratch.com/neural-network-tutorial/
#            https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/
#            https://www.jeremyjordan.me/convolutional-neural-networks/
#            https://victorzhou.com/series/neural-networks-from-scratch/

"""
数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
正确率：81.72%（二分类）
运行时长：78.6s
"""

import numpy as np
import time
from DataProcess import DataProcess

np.random.seed(0)


def sigmoid(array):
    """
    计算sigmoid函数，sigmoid(x) = 1/(1+ exp(-x))
    :param array: numpy.ndarray类型的输入
    :return: 返回sigmoid计算结果
    """
    return 1 / (1 + np.exp(array))


def der_sigmoid(array):
    """sigmoid函数的导数，der_sigmoid(x) = sigmoid(x) * (1-sigmoid(x))
    :param array: numpy.ndarray类型的输入
    :return: 返回sigmoid的求导结果
    """
    return sigmoid(array) * (1 - sigmoid(array))


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


def softmax(array):
    """
    用于计算array的每一行softmax。利用softmax函数的性质: softmax(x) = softmax(x + c)
    :param array: numpy.ndarray类型输入
    :return: 返回softmax计算结果
    """
    orig_shape = array.shape
    # 根据输入类型是矩阵还是向量分别计算softmax
    if len(array.shape) > 1:
        # 矩阵
        tmp = np.max(array, axis=1)  # 得到每行的最大值，用于缩放每行的元素，避免溢出
        array -= tmp.reshape((array.shape[0], 1))  # 利用性质缩放元素
        array = np.exp(array)  # 计算所有值的指数
        tmp = np.sum(array, axis=1)  # 每行求和
        array /= tmp.reshape((array.shape[0], 1))  # 求softmax
    else:
        # 向量
        tmp = np.max(array)  # 得到最大值
        array -= tmp  # 利用最大值缩放数据
        array = np.exp(array)  # 对所有元素求指数
        tmp = np.sum(array)  # 求元素和
        array /= tmp  # 求softmax
    return array


def cross_entropy(x, y):
    x_softmax = [softmax(x[i]) for i in range(len(x))]
    x_log = [np.log(x_softmax[i][y[i]]) for i in range(len(y))]
    loss = - np.sum(x_log) / len(y)
    return loss


def der_cross_entropy_softmax(act_array, pre_array):
    """
    用于计算经softmax后的Cross Entropy的梯度
    :param act_array: 实际标签的One-Hot变量，维度为：n_sample*n_class
    :param pre_array: 预测值经softmax后的概率分布，维度为：n_sample*n_class
    :return: 返回经softmax后的Cross Entropy的梯度，维度为：n_sample*n_class
    """
    y_act = act_array
    y_hat = pre_array
    return y_hat - y_act


def one_hot(num_class, label_array):
    """
    用于将样本标签转换为One_Hot变量
    :param num_class: 样本标签的类别数量，维度为：1
    :param label_array: numpy.ndarray类型的标签，维度为：n_sample*1
    :return: 返回样本标签的OneHot变量，维度为：n_sample * n_class
    """
    return np.eye(num_class)[label_array]


class DeepNeuralNetwork(object):
    def __init__(self, x_input, y_out, num_hidden, num_class):
        print("Initialization DeepNeuralNetwork's parameters")
        self.num_sample = x_input.shape[0]
        self.num_input = x_input.shape[1]
        self.num_hidden = num_hidden
        self.num_out = num_class
        self.w1 = np.random.rand(self.num_input, self.num_hidden)
        self.b1 = np.random.rand(self.num_sample, 1)
        self.w2 = np.random.rand(self.num_hidden, self.num_out)
        self.b2 = np.random.rand(self.num_sample, 1)

    def forward(self, feature_array):
        print("Forward Propagation")
        A1 = np.dot(feature_array, self.w1) + self.b1
        # Z1 = sigmoid(A1)
        Z1 = relu(A1)
        A2 = np.dot(Z1, self.w2) + self.b2
        Z2 = softmax(A2)

        cache = {
            "A1": A1,
            "Z1": Z1,
            "A2": A2,
            "Z2": Z2}

        return Z2, cache

    def backward(self, feature_array, label_array, learning_rate=0.1):
        print("Backward Propagation")
        Z2, cache = self.forward(feature_array)
        label_array_onehot = one_hot(self.num_out, label_array)

        # 获取Z1和Z2
        Z2 = cache['Z2']
        A2 = cache['A2']
        Z1 = cache['Z1']
        A1 = cache['A1']
        print("Z2 shape", Z2.shape)
        print("A2 shape", A2.shape)
        print("Z1 shape", Z1.shape)
        print("A1 shape", A1.shape)

        dL_A2 = der_cross_entropy_softmax(label_array_onehot, Z2)
        print("dL_A2 shape", dL_A2.shape)
        dW2 = 1 / self.num_sample * np.dot(Z1.T, dL_A2)
        db2 = 1 / self.num_sample * np.sum(dL_A2, axis=1, keepdims=True)

        # dL_A1 = np.dot(dL_A2, self.w2.T) * der_sigmoid(A1)
        dL_A1 = np.dot(dL_A2, self.w2.T) * der_relu(A1)
        print("dL_A1 shape", dL_A1.shape)
        dW1 = 1 / self.num_sample * np.dot(feature_array.T, dL_A1)
        db1 = 1 / self.num_sample * np.sum(dL_A1, axis=1, keepdims=True)

        # 参数更新
        self.w1 -= dW1 * learning_rate
        self.b1 -= db1 * learning_rate
        self.w2 -= dW2 * learning_rate
        self.b2 -= db2 * learning_rate

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2,
                 "dL_A2": dL_A2,
                 "dL_A1": dL_A1}
        return grads

    def update_parameters(self, grads, learning_rate=1.2):
        # 获取梯度
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']

        # 参数更新
        self.w1 -= dW1 * learning_rate
        self.b1 -= db1 * learning_rate
        self.w2 -= dW2 * learning_rate
        self.b2 -= db2 * learning_rate


if __name__ == '__main__':
    start = time.time()
    # 加载数据
    # 获取训练集及标签
    trainData, trainLabel = DataProcess.load_data('./data/Mnist/mnist_train.csv', binary_classification=False)
    # 获取测试集及标签
    # testData, testLabel = DataProcess.load_data('./data/Mnist/mnist_test.csv',binary_classification=False)
    # 实例化感知机，输入特征为4
    dnn = DeepNeuralNetwork(trainData, trainLabel, 88, 10)
    for i in range(500):
        grads = dnn.backward(trainData, trainLabel)
        # dnn.update_parameters(grads, learning_rate=0.01)
        # print(dnn.w1.max())
        label_pre, cache_test = dnn.forward(trainData)
        # loss_ce = -np.mean(np.sum(np.multiply(one_hot(10, trainLabel), np.log(label_pre)), axis=1))
        loss_ce = cross_entropy(label_pre, trainLabel)
        acc = np.sum(trainLabel == np.argmax(label_pre, axis=1)) / trainLabel.shape[0]
        if not i % 50:
            print("Iter", i, "Loss", loss_ce, "Acc", acc)
    end = time.time()
    # 显示用时时长
    print('time span:', end - start)
