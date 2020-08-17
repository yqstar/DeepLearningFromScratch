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
from Utils import Utils

np.random.seed(0)


class DeepNeuralNetwork(object):
    def __init__(self, input_x, hidden, out_y):
        self.input_x = input_x
        self.hidden = hidden
        self.out_y = out_y

    def forward(self):
        pass

    def backward(self):
        pass


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.layer1 = Utils.sigmoid(np.dot(self.input, self.weights1))
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def forward(self):
        self.layer1 = Utils.sigmoid(np.dot(self.input, self.weights1))
        self.output = Utils.sigmoid(np.dot(self.layer1, self.weights2))

    def backward(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * Utils.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (
                np.dot(2 * (self.y - self.output) * Utils.sigmoid_derivative(self.output),
                       self.weights2.T) * Utils.sigmoid_derivative(self.layer1)))
        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == '__main__':
    start = time.time()
    # 加载数据
    # 获取训练集及标签
    trainData, trainLabel = DataProcess.load_data('./data/Mnist/mnist_train.csv')
    # 获取测试集及标签
    testData, testLabel = DataProcess.load_data('./data/Mnist/mnist_test.csv')
    # 实例化感知机，输入特征为4
    dnn = NeuralNetwork(trainData, trainLabel)
    for i in range(100):
        dnn.backward()
        acc = np.sum(dnn.y == dnn.output) / dnn.y.shape[0]
        print(acc)
    # # 训练模型，获得计算参数
    # weight, bias = dnn.forward(trainData, trainLabel, iteration=1000)
    # # 预测结果
    # y_pre = dnn.backward(testData)
    end = time.time()
    # 显示用时时长
    print('time span:', end - start)
