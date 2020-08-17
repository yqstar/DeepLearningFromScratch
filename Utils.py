import numpy as np


class Utils(object):
    @staticmethod
    def sigmoid(array):
        return 1 / (1 + np.exp(array))

    @staticmethod
    def sigmoid_derivative(array):
        sigmoid_value = 1 / (1 + np.exp(array))
        return np.multiply(sigmoid_value, 1 - sigmoid_value)
