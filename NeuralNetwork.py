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
