# -*- coding:utf-8 -*-
# Author:AndrewYq
# Date:2020-08-16
# Email:hfyqstar@163.com

'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
正确率：81.72%（二分类）
运行时长：78.6s
'''

# https://mlfromscratch.com/neural-network-tutorial/
# https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/
# https://www.jeremyjordan.me/convolutional-neural-networks/
# https://victorzhou.com/series/neural-networks-from-scratch/

from sklearn.datasets import fetch_openml
# from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x/255).astype('float32')
# y = to_categorical(y)

# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)