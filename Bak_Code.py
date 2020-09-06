import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import numpy as np
import time
from DataProcess import DataProcess
from Utils import Utils


def create_dataset():
    np.random.seed(1)
    m = 400  # 数据量
    N = int(m / 2)  # 每个标签的实例数
    D = 2  # 数据维度
    X = np.zeros((m, D))  # 数据矩阵
    Y = np.zeros((m, 1), dtype='uint8')  # 标签维度
    a = 4

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def layer_sizes(X, Y):
    n_x = X.shape[0]  # 输入层大小
    n_h = 20  # 隐藏层大小
    n_y = Y.shape[0]  # 输出层大小
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def forward_propagation(X, parameters):
    # 获取各参数初始值
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # 执行前向计算
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    print(A2.shape)
    print(X.shape[1])
    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    # 训练样本量
    m = Y.shape[1]
    # 计算交叉熵损失
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -1 / m * np.sum(logprobs)
    # 维度压缩
    cost = np.squeeze(cost)

    assert (isinstance(cost, float))
    return cost


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    # 获取W1和W2
    W1 = parameters['W1']
    W2 = parameters['W2']
    # 获取A1和A2
    A1 = cache['A1']
    A2 = cache['A2']
    # 执行反向传播
    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    # 获取参数
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # 获取梯度
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    # 参数更新
    W1 -= dW1 * learning_rate
    b1 -= db1 * learning_rate
    W2 -= dW2 * learning_rate
    b2 -= db2 * learning_rate

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    # 初始化模型参数
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # 梯度下降和参数更新循环
    for i in range(0, num_iterations):
        # 前向传播计算
        A2, cache = forward_propagation(X, parameters)
        # 计算当前损失
        cost = compute_cost(A2, Y, parameters)
        # 反向传播
        grads = backward_propagation(parameters, cache, X, Y)
        # 参数更新
        parameters = update_parameters(parameters, grads, learning_rate=1.2)
        # 打印损失
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions


if __name__ == '__main__':
    start = time.time()
    # 加载数据
    # 获取训练集及标签
    trainData, trainLabel = DataProcess.load_data('./data/Mnist/mnist_train.csv')
    # 获取测试集及标签
    testData, testLabel = DataProcess.load_data('./data/Mnist/mnist_test.csv')

    X = trainData.T[:,0:50]
    Y = trainLabel.T[0:50]
    X, Y = create_dataset()
    # 实例化感知机，输入特征为4
    parameters = nn_model(X, Y, n_h=20, num_iterations=10000, print_cost=True)
    predictions = predict(parameters, X)
    print('Accuracy: %d' % float((np.dot(Y, predictions.T) +
                                  np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
    # pre = dnn.forward(trainData)
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
    # X, Y = create_dataset()
    # # plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral)
    # parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)
    # predictions = predict(parameters, X)
    # print ('Accuracy: %d' % float((np.dot(Y,predictions.T) +
    #       np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

# def plot_decision_boundary(model, X, y):
#     # Set min and max values and give it some padding
#     x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
#     y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
#     h = 0.01
#     # Generate a grid of points with distance h between them
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     # Predict the function value for the whole grid
#     Z = model(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     # Plot the contour and training examples
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
#     plt.ylabel('x2')
#     plt.xlabel('x1')
#     plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
#
#
# plt.figure(figsize=(16, 32))
# hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20]
# for i, n_h in enumerate(hidden_layer_sizes):
#     plt.subplot(5, 2, i+1)
#     plt.title('Hidden Layer of size %d' % n_h)
#     parameters = nn_model(X, Y, n_h, num_iterations = 5000)
#     plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0])
#     predictions = predict(parameters, X)
#     accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
#     print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

# # !/usr/bin/env python
# # -*- coding: UTF-8 -*-
#
#
# import numpy as np
#
#
# # from cnn import element_wise_op
# # from activators import ReluActivator, IdentityActivator
#
# # 对numpy数组进行element wise操作
# def element_wise_op(array, op):
#     for i in np.nditer(array,
#                        op_flags=['readwrite']):
#         i[...] = op(i)
#
#
# import numpy as np
#
#
# class ReluActivator(object):
#     def forward(self, weighted_input):
#         # return weighted_input
#         return max(0, weighted_input)
#
#     def backward(self, output):
#         return 1 if output > 0 else 0
#
#
# class IdentityActivator(object):
#     def forward(self, weighted_input):
#         return weighted_input
#
#     def backward(self, output):
#         return 1
#
#
# class SigmoidActivator(object):
#     def forward(self, weighted_input):
#         return 1.0 / (1.0 + np.exp(-weighted_input))
#
#     def backward(self, output):
#         return output * (1 - output)
#
#
# class TanhActivator(object):
#     def forward(self, weighted_input):
#         return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0
#
#     def backward(self, output):
#         return 1 - output * output
#
#
# class RecurrentLayer(object):
#     def __init__(self, input_width, state_width,
#                  activator, learning_rate):
#         self.input_width = input_width
#         self.state_width = state_width
#         self.activator = activator
#         self.learning_rate = learning_rate
#         self.times = 0  # 当前时刻初始化为t0
#         self.state_list = []  # 保存各个时刻的state
#         self.state_list.append(np.zeros(
#             (state_width, 1)))  # 初始化s0
#         self.U = np.random.uniform(-1e-4, 1e-4,
#                                    (state_width, input_width))  # 初始化U
#         self.W = np.random.uniform(-1e-4, 1e-4,
#                                    (state_width, state_width))  # 初始化W
#
#     def forward(self, input_array):
#         '''
#         根据『式2』进行前向计算
#         '''
#         self.times += 1
#         state = (np.dot(self.U, input_array) +
#                  np.dot(self.W, self.state_list[-1]))
#         element_wise_op(state, self.activator.forward)
#         self.state_list.append(state)
#
#     def backward(self, sensitivity_array,
#                  activator):
#         '''
#         实现BPTT算法
#         '''
#         self.calc_delta(sensitivity_array, activator)
#         self.calc_gradient()
#
#     def update(self):
#         '''
#         按照梯度下降，更新权重
#         '''
#         self.W -= self.learning_rate * self.gradient
#
#     def calc_delta(self, sensitivity_array, activator):
#         self.delta_list = []  # 用来保存各个时刻的误差项
#         for i in range(self.times):
#             self.delta_list.append(np.zeros(
#                 (self.state_width, 1)))
#         self.delta_list.append(sensitivity_array)
#         # 迭代计算每个时刻的误差项
#         for k in range(self.times - 1, 0, -1):
#             self.calc_delta_k(k, activator)
#
#     def calc_delta_k(self, k, activator):
#         '''
#         根据k+1时刻的delta计算k时刻的delta
#         '''
#         state = self.state_list[k + 1].copy()
#         element_wise_op(self.state_list[k + 1],
#                         activator.backward)
#         self.delta_list[k] = np.dot(
#             np.dot(self.delta_list[k + 1].T, self.W),
#             np.diag(state[:, 0])).T
#
#     def calc_gradient(self):
#         self.gradient_list = []  # 保存各个时刻的权重梯度
#         for t in range(self.times + 1):
#             self.gradient_list.append(np.zeros(
#                 (self.state_width, self.state_width)))
#         for t in range(self.times, 0, -1):
#             self.calc_gradient_t(t)
#         # 实际的梯度是各个时刻梯度之和
#         self.gradient = reduce(
#             lambda a, b: a + b, self.gradient_list,
#             self.gradient_list[0])  # [0]被初始化为0且没有被修改过
#
#     def calc_gradient_t(self, t):
#         '''
#         计算每个时刻t权重的梯度
#         '''
#         gradient = np.dot(self.delta_list[t],
#                           self.state_list[t - 1].T)
#         self.gradient_list[t] = gradient
#
#     def reset_state(self):
#         self.times = 0  # 当前时刻初始化为t0
#         self.state_list = []  # 保存各个时刻的state
#         self.state_list.append(np.zeros(
#             (self.state_width, 1)))  # 初始化s0
#
#
# def data_set():
#     x = [np.array([[1], [2], [3]]),
#          np.array([[2], [3], [4]])]
#     d = np.array([[1], [2]])
#     return x, d
#
#
# def gradient_check():
#     '''
#     梯度检查
#     '''
#     # 设计一个误差函数，取所有节点输出项之和
#     error_function = lambda o: o.sum()
#
#     rl = RecurrentLayer(3, 2, IdentityActivator(), 1e-3)
#
#     # 计算forward值
#     x, d = data_set()
#     rl.forward(x[0])
#     rl.forward(x[1])
#
#     # 求取sensitivity map
#     sensitivity_array = np.ones(rl.state_list[-1].shape,
#                                 dtype=np.float64)
#     # 计算梯度
#     rl.backward(sensitivity_array, IdentityActivator())
#
#     # 检查梯度
#     epsilon = 10e-4
#     for i in range(rl.W.shape[0]):
#         for j in range(rl.W.shape[1]):
#             rl.W[i, j] += epsilon
#             rl.reset_state()
#             rl.forward(x[0])
#             rl.forward(x[1])
#             err1 = error_function(rl.state_list[-1])
#             rl.W[i, j] -= 2 * epsilon
#             rl.reset_state()
#             rl.forward(x[0])
#             rl.forward(x[1])
#             err2 = error_function(rl.state_list[-1])
#             expect_grad = (err1 - err2) / (2 * epsilon)
#             rl.W[i, j] += epsilon
#             print('weights(%d,%d): expected - actural %f - %f' % (
#                 i, j, expect_grad, rl.gradient[i, j]))
#
#
# def test():
#     l = RecurrentLayer(3, 2, ReluActivator(), 1e-3)
#     x, d = data_set()
#     l.forward(x[0])
#     l.forward(x[1])
#     l.backward(d, ReluActivator())
#     return l
