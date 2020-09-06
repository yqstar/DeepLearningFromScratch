# http://songhuiming.github.io/pages/2017/08/12/build-neural-network-from-scratch/
# https://songhuiming.github.io/pages/2017/08/20/build-recurrent-neural-network-from-scratch/
# https://songhuiming.github.io/category/python.html

# http://songhuiming.github.io/pages/2017/08/12/build-neural-network-from-scratch/
# https://songhuiming.github.io/pages/2017/08/20/build-recurrent-neural-network-from-scratch/
# https://songhuiming.github.io/category/python.html
# https://zybuluo.com/hanbingtao/note/541458

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime


def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)


def save_model_parameters_numpy(outfile, model):
    """
    Save model's parameters with numpy
    :param outfile: stored file path
    :param model: RNN's model
    :return: None
    """
    U, V, W = model.U, model.V, model.W
    np.savez(outfile, U=U, V=V, W=W)
    print("Saved model parameters to %s." % outfile)


def load_model_parameters_numpy(path, model):
    """
    Load model's parameters with numpy
    :param path: stored file path
    :param model: RNN's model
    :return: None
    """
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U = U
    model.V = V
    model.W = W
    print("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1]))


class RecurrentNeuralNetworks:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim),
                                   (hidden_dim, word_dim))  # shape:hidden_dim*word_dim
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim),
                                   (word_dim, hidden_dim))  # shape:word_dim*hidden_dim
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim),
                                   (hidden_dim, hidden_dim))  # shape:hidden_dim*hidden_dim

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                        np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print("+h Loss: %f" % gradplus)
                    print("-h Loss: %f" % gradminus)
                    print("Estimated_gradient: %f" % estimated_gradient)
                    print("Backpropagation gradient: %f" % backprop_gradient)
                    print("Relative Error: %f" % relative_error)
                    return
                it.iternext()
            print("Gradient check for parameter %s passed." % (pname))

    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    # Outer SGD Loop
    # - model: The RNN model instance
    # - X_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - nepoch: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs
    def train_with_sgd(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
                # ADDED! Saving model oarameters
                save_model_parameters_numpy("./data/rnn-numpy-%d-%d-%s.npz" % (self.hidden_dim, self.word_dim, time),
                                            self)
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1


def read_sentences_from_csv(path, sentence_start_token, sentence_end_token):
    print("Reading CSV file...")

    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)
            # Split full comments into sentences
            # sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
            # Append SENTENCE_START and SENTENCE_END
            sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    # print(type(reader))
    # for row in reader:
    #     print(row)
    #     # Split full comments into sentences
    #     sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    #     # Append SENTENCE_START and SENTENCE_END
    #     sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    # with open(path) as f:
    #     reader = csv.reader(f, skipinitialspace=True)
    #     # reader.next()
    #     # reader = next(reader)
    #     # Split full comments into sentences
    #     sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    #     # Append SENTENCE_START and SENTENCE_END
    #     sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))
    return sentences


def build_vocabulary(tokenized_sentences, vocabulary_size, unknown_token):
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    return [index_to_word, word_to_index]





# from utils import *
# from rnn_theano import RNNTheano
# from rnn_numpy import RNNNumpy

class Config:
    _DATASET_FILE = os.environ.get('DATASET_FILE', './data/small-dataset.csv')
    _MODEL_FILE = os.environ.get('MODEL_FILE', './data/small_rnn_model.npz')

    _VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '20'))
    _UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
    _SENTENCE_START_TOKEN = "SENTENCE_START"
    _SENTENCE_END_TOKEN = "SENTENCE_END"

    _HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '20'))
    _LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
    _NEPOCH = int(os.environ.get('NEPOCH', '10'))


# Read the data and append SENTENCE_START and SENTENCE_END tokens
sentences = read_sentences_from_csv(Config._DATASET_FILE, Config._SENTENCE_START_TOKEN, Config._SENTENCE_END_TOKEN)

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

index_to_word, word_to_index = build_vocabulary(tokenized_sentences, Config._VOCABULARY_SIZE, Config._UNKNOWN_TOKEN)

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else Config._UNKNOWN_TOKEN for w in sent]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])


def train_numpy():
    model = RecurrentNeuralNetworks(Config._VOCABULARY_SIZE, hidden_dim=Config._HIDDEN_DIM)
    t1 = time.time()
    model.sgd_step(X_train[10], y_train[10], Config._LEARNING_RATE)
    t2 = time.time()
    print("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))

    model.train_with_sgd(X_train, y_train, nepoch=Config._NEPOCH, learning_rate=Config._LEARNING_RATE)
    # train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)

    if Config._MODEL_FILE != None:
        print("start saving model...")
        save_model_parameters_numpy(Config._MODEL_FILE, model)
        print("model saved!")



# train_numpy()

# def train_theano():
#     model = RNNTheano(Config._VOCABULARY_SIZE, hidden_dim=Config._HIDDEN_DIM)
#     t1 = time.time()
#     model.sgd_step(X_train[10], y_train[10], Config._LEARNING_RATE)
#     t2 = time.time()
#     print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
#
#     model.train_with_sgd(X_train, y_train, nepoch=Config._NEPOCH, learning_rate=Config._LEARNING_RATE)
#
#     if Config._MODEL_FILE != None:
#         print "start saving model..."
#         save_model_parameters_theano(Config._MODEL_FILE, model)
#         print "model saved!"

# train_theano()

# from functools import reduce
#
# import numpy as np
#
#
# def relu(array):
#     return np.exp(array) / (1 + np.exp(array))
#
#
# def der_relu(array):
#     pass
#
#
# class RNN(object):
#     def __init__(self, num_class):
#         self.num_class = num_class
#         pass
#
#     def forward(self):
#         pass
#
#     def backward(self):
#         pass
#
#     def gradient_check(self):
#         pass
#
#
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

