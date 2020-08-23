# coding=utf-8
# Author:AndrewYq
# Date:2020-08-15
# Email:hfyqstar@163.com

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


class Perceptron(object):
    def __init__(self, num_input):
        self.num_input = num_input
        # 初始化Weights(w:b)
        self.weight = np.random.rand(num_input)
        self.bias = np.random.rand(1)

    def __str__(self):
        return "Weight:{0},Bias:{1}".format(self.weight, self.bias)

    # 输入数据，输出结果
    def predict(self, feature_array):
        # bias_array = np.ones(len(input_array))
        # feature_array = np.column_stack((input_array, bias_array))
        tmp = np.dot(feature_array, self.weight) + self.bias
        tmp = np.where(tmp > 0, 1, -1)
        return tmp

    def fit(self, feature_array, label_array, lr=0.01, iteration=50):
        for iter in range(iteration):
            n_sample = len(feature_array)
            for i in range(n_sample):
                sample_flag = -label_array[i] * (np.dot(feature_array[i], self.weight) + self.bias)
                if sample_flag > 0:
                    self.weight = self.weight + lr * label_array[i] * feature_array[i]
                    self.bias = self.bias + lr * label_array[i]
            loss_flag = np.multiply(-label_array, (np.dot(feature_array, self.weight) + self.bias))
            loss = np.sum(loss_flag[loss_flag > 0])
            acc = 1.0 - len(loss_flag[loss_flag > 0]) / len(loss_flag)
            print("Iteration:", iter, "Loss:", loss, "Acc:", acc)
        return self.weight, self.bias

    def cal_acc(self, y_acc, y_hat):
        pass


if __name__ == '__main__':
    start = time.time()
    # 加载数据
    # 获取训练集及标签
    trainData, trainLabel = DataProcess.load_data('./data/Mnist/mnist_train.csv')
    # 获取测试集及标签
    testData, testLabel = DataProcess.load_data('./data/Mnist/mnist_test.csv')
    # 实例化感知机，输入特征为4
    mlp = Perceptron(num_input=784)
    # 训练模型，获得计算参数
    weight, bias = mlp.fit(trainData, trainLabel, iteration=1000)
    # 预测结果
    y_pre = mlp.predict(testData)
    end = time.time()
    # 显示用时时长
    print('time span:', end - start)
