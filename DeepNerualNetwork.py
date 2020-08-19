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


def softmax(array):
    """Compute the softmax of vector array."""
    exps = np.exp(array)
    return exps / np.sum(exps)


def cross_entropy(X, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector.
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    p = softmax(X)
    # We use multidimensional array indexing to extract
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss


def delta_cross_entropy(X, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector.
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad


def onehot(num_class, label_array):
    return np.eye(num_class)[label_array]
    # max_label = np.max(label_array)
    # if max_label < num_class:
    #     tmp = np.eye(num_class)[label_array]
    # else:
    #     print("标签数据有误")
    #     pass
    # return tmp


class DeepNeuralNetwork(object):
    def __init__(self, input_x, hidden, out_y, num_class):
        self.input_x = input_x
        self.hidden = hidden
        self.out_y = out_y
        self.w1 = np.random.rand(self.input_x, self.hidden)
        self.b1 = np.random.rand(60000, 1)
        self.w2 = np.random.rand(self.hidden, self.out_y)
        self.b2 = np.random.rand(60000, 1)
        self.num_class = num_class

    def forward(self, feature_array):
        hidden_layer = Utils.sigmoid(np.dot(feature_array, self.w1) + self.b1)
        out_layer = Utils.softmax(np.dot(hidden_layer, self.w2) + self.b2)
        return out_layer, np.argmax(out_layer, axis=1)

    def backward(self, feature_array, label_array, lr=0.001, iteration=50):
        label_onehot = onehot(self.num_class, label_array)
        n_sample = len(feature_array)
        out_y = np.dot(Utils.sigmoid(np.dot(feature_array, self.w1) + self.b1), self.w2) + self.b2
        out_softmax = Utils.softmax(out_y)
        loss_ce = np.dot(label_onehot, np.log(out_softmax))


        grad_dw2 = np.dot(delta_cross_entropy(out_y,label_array),np.dot(feature_array, self.w1) + self.b1)
        grad_db2 = delta_cross_entropy(out_y,label_array)



        grad_dw1 = np.dot(grad_dw2,)
        grad_db1 = np.dot(grad_db2,)

        self.w1 = self.w1 - lr * grad_dw1
        self.b1 = self.b1 - lr * grad_db1
        self.w2 = self.w2 - lr * grad_dw2
        self.b2 = self.b2 - lr * grad_db2





        return True

    @staticmethod
    def cross_entropy(p, q):
        """
        计算多分类的交叉熵
        :param p: 分类的实际标签值
        :param q: SoftMax后的概率值
        :return: 交叉熵，多用于多分类的损失函数
        """
        return np.dot(p, np.log(q))


if __name__ == '__main__':
    start = time.time()
    # 加载数据
    # 获取训练集及标签
    trainData, trainLabel = DataProcess.load_data('./data/Mnist/mnist_train.csv')
    # 获取测试集及标签
    testData, testLabel = DataProcess.load_data('./data/Mnist/mnist_test.csv')
    # 实例化感知机，输入特征为4
    dnn = DeepNeuralNetwork(28 * 28, 64, 10)
    pre = dnn.forward(trainData)
    # for i in range(100):
    #     dnn.backward()
    #     acc = np.sum(dnn.y == dnn.output) / dnn.y.shape[0]
    #     print(acc)
    # # 训练模型，获得计算参数
    # weight, bias = dnn.forward(trainData, trainLabel, iteration=1000)
    # # 预测结果
    # y_pre = dnn.backward(testData)
    end = time.time()
    # 显示用时时长
    print('time span:', end - start)

#
#
# class NeuralNetwork:
#     def __init__(self, x, y):
#         self.input = x
#         self.weights1 = np.random.rand(self.input.shape[1], 4)
#         self.layer1 = Utils.sigmoid(np.dot(self.input, self.weights1))
#         self.weights2 = np.random.rand(4, 1)
#         self.y = y
#         self.output = np.zeros(self.y.shape)
#
#     def forward(self):
#         self.layer1 = Utils.sigmoid(np.dot(self.input, self.weights1))
#         self.output = Utils.sigmoid(np.dot(self.layer1, self.weights2))
#
#     def backward(self):
#         # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
#         d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * Utils.sigmoid_derivative(self.output)))
#         d_weights1 = np.dot(self.input.T, (
#                 np.dot(2 * (self.y - self.output) * Utils.sigmoid_derivative(self.output),
#                        self.weights2.T) * Utils.sigmoid_derivative(self.layer1)))
#         # update the weights with the derivative (slope) of the loss function
#         self.weights1 += d_weights1
#         self.weights2 += d_weights2
#
# from math import exp
# from random import seed
# from random import random
#
#
# # Initialize a network
# def initialize_network(n_inputs, n_hidden, n_outputs):
#     network = list()
#     hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
#     network.append(hidden_layer)
#     output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
#     network.append(output_layer)
#     return network
#
#
# # Calculate neuron activation for an input
# def activate(weights, inputs):
#     activation = weights[-1]
#     for i in range(len(weights) - 1):
#         activation += weights[i] * inputs[i]
#     return activation
#
#
# # Transfer neuron activation
# def transfer(activation):
#     return 1.0 / (1.0 + exp(-activation))
#
#
# # Forward propagate input to a network output
# def forward_propagate(network, row):
#     inputs = row
#     for layer in network:
#         new_inputs = []
#         for neuron in layer:
#             activation = activate(neuron['weights'], inputs)
#             neuron['output'] = transfer(activation)
#             new_inputs.append(neuron['output'])
#         inputs = new_inputs
#     return inputs
#
#
# # Calculate the derivative of an neuron output
# def transfer_derivative(output):
#     return output * (1.0 - output)
#
#
# # Backpropagate error and store in neurons
# def backward_propagate_error(network, expected):
#     for i in reversed(range(len(network))):
#         layer = network[i]
#         errors = list()
#         if i != len(network) - 1:
#             for j in range(len(layer)):
#                 error = 0.0
#                 for neuron in network[i + 1]:
#                     error += (neuron['weights'][j] * neuron['delta'])
#                 errors.append(error)
#         else:
#             for j in range(len(layer)):
#                 neuron = layer[j]
#                 errors.append(expected[j] - neuron['output'])
#         for j in range(len(layer)):
#             neuron = layer[j]
#             neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
#
#
# # Update network weights with error
# def update_weights(network, row, l_rate):
#     for i in range(len(network)):
#         inputs = row[:-1]
#         if i != 0:
#             inputs = [neuron['output'] for neuron in network[i - 1]]
#         for neuron in network[i]:
#             for j in range(len(inputs)):
#                 neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
#             neuron['weights'][-1] += l_rate * neuron['delta']
#
#
# # Train a network for a fixed number of epochs
# def train_network(network, train, l_rate, n_epoch, n_outputs):
#     for epoch in range(n_epoch):
#         sum_error = 0
#         for row in train:
#             outputs = forward_propagate(network, row)
#             expected = [0 for i in range(n_outputs)]
#             expected[row[-1]] = 1
#             sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
#             backward_propagate_error(network, expected)
#             update_weights(network, row, l_rate)
#         print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# Test training backprop algorithm
# seed(1)
# dataset = [[2.7810836, 2.550537003, 0],
#            [1.465489372, 2.362125076, 0],
#            [3.396561688, 4.400293529, 0],
#            [1.38807019, 1.850220317, 0],
#            [3.06407232, 3.005305973, 0],
#            [7.627531214, 2.759262235, 1],
#            [5.332441248, 2.088626775, 1],
#            [6.922596716, 1.77106367, 1],
#            [8.675418651, -0.242068655, 1],
#            [7.673756466, 3.508563011, 1]]
# n_inputs = len(dataset[0]) - 1
# n_outputs = len(set([row[-1] for row in dataset]))
# network = initialize_network(n_inputs, 2, n_outputs)
# train_network(network, dataset, 0.5, 20, n_outputs)
# for layer in network:
#     print(layer)
