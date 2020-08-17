import numpy as np
from sklearn import datasets


class DataProcess(object):
    @staticmethod
    def load_data(file_name):
        '''
        加载Mnist数据集
        :param file_name:要加载的数据集路径
        :return: list形式的数据集及标记
        '''
        print('start to read data')
        # 存放数据及标记的list
        dataArr = []
        labelArr = []
        # 打开文件
        fr = open(file_name, 'r')
        # 将文件按行读取
        for line in fr.readlines():
            # 对每一行数据按切割福','进行切割，返回字段列表
            curLine = line.strip().split(',')

            # Mnsit有0-9是个标记，由于是二分类任务，所以将>=5的作为1，<5为-1
            if int(curLine[0]) >= 5:
                labelArr.append(1)
            else:
                labelArr.append(-1)
            # 存放标记
            # [int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一哥元素（标记）外将所有元素转换成int类型
            # [int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)
            dataArr.append([int(num) / 255 for num in curLine[1:]])

        # 返回data和label
        return np.array(dataArr), np.array(labelArr)

    @staticmethod
    def load_data_test():
        '''
        加载Iris数据集(The iris dataset is a classic and very easy multi-class classification dataset.)
        :return:ndarray的数据集及标记
        '''
        feature_array, target_array = datasets.load_iris(return_X_y=True)
        # Iris有0-2三个标记，由于是二分类任务，所以将0,1的作为-1，2为1
        target_array = np.piecewise(target_array, [target_array == 0, target_array == 1, target_array == 2],
                                    [-1, -1, 1])
        return feature_array, target_array
