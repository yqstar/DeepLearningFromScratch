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


# def softmax(array):
#     """Compute the softmax of vector array."""
#     exps = np.exp(array)
#     return exps / np.sum(exps)

def softmax(X):
    X_exp = np.exp(X)
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition  # 这里应用了广播机制。


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


class DeepNeuralNetwork(object):
    def __init__(self, x_input, y_out, num_hidden, num_class):
        self.num_sample = x_input.shape[0]
        self.num_input = x_input.shape[1]
        self.num_hidden = num_hidden
        self.num_out = num_class
        self.w1 = np.random.rand(self.num_input, self.num_hidden)
        self.b1 = np.random.rand(self.num_sample, 1)
        self.w2 = np.random.rand(self.num_hidden, self.num_out)
        self.b2 = np.random.rand(self.num_sample, 1)
        print("test")

    def forward(self, feature_array):
        A1 = np.dot(feature_array, self.w1) + self.b1
        Z1 = sigmoid(A1)
        A2 = np.dot(Z1, self.w2) + self.b2
        Z2 = softmax(A2)

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}
        return Z2, cache

    def backward(self, feature_array, label_array, learning_rate=0.1):
        # m = feature_array.shape[1]
        # print(m)
        Z2, cache = self.forward(feature_array)
        # 获取Z1和Z2
        Z1 = cache['Z1']
        Z2 = cache['Z2']
        A1 = cache['A1']
        A2 = cache['A2']
        print("Z1 shape", Z1.shape)
        print("Z2 shape", Z2.shape)
        print("A1 shape", A1.shape)
        print("A2 shape", A2.shape)

        label_onehot = onehot(self.num_out, label_array)
        dL_A2 = der_softmax_cross_entropy(label_onehot, Z2)
        print("dL_A2 shape", dL_A2.shape)
        # dW2 = 1/m * np.dot(Z1.T, dL_A2)
        m = dL_A2.shape[1]
        dW2 = np.dot(Z1.T, dL_A2)
        # db2 = 1 / m * np.sum(dL_A2, axis=1, keepdims=True)
        db2 = np.sum(dL_A2, axis=1, keepdims=True)

        # print("w2 shape", self.w2.shape)
        dL_A1 = np.dot(dL_A2, self.w2.T) * der_sigmoid(A1)
        print("dL_A1 shape", dL_A1.shape)
        n = dL_A1.shape[1]
        # dW1 = 1/m * np.dot(feature_array.T, dL_A1)
        dW1 = np.dot(feature_array.T, dL_A1)
        # db1 = 1 / n * np.sum(dL_A1, axis=1, keepdims=True)
        db1 = np.sum(dL_A1, axis=1, keepdims=True)

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
    dnn = DeepNeuralNetwork(trainData, trainLabel, 64, 10)
    for i in range(1):
        grads = dnn.backward(trainData, trainLabel)
        # dnn.update_parameters(grads, learning_rate=0.01)
        print(dnn.w1.max())
        label_pre, _ = dnn.forward(trainData)
        loss_ce = -np.mean(np.sum(np.multiply(onehot(10, trainLabel), np.log(label_pre)), axis=1))
        acc = np.sum(trainLabel == np.argmax(label_pre, axis=1)) / trainLabel.shape[0]
        # if not i % 50:
        print("Iter", i, "Loss", loss_ce, "Acc", acc)

    end = time.time()
    # 显示用时时长
    print('time span:', end - start)
