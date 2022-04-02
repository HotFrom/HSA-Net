"""
Created on 2018/10/21 by Chunhui Yin(yinchunhui.ahu@gmail.com).
Description:Loading the data.

"""
import sys
from time import time
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
class DataSet(object):

    def __init__(self, dataType, density):

        self.dataType, self.density = dataType, density
        self.data, self.shape = self.getData()
        self.train, self.test = self.getTrainTest()

    def getData(self):
        self.start = time()
        sys.stdout.write('\rLoading data...')
        if self.dataType == 'LDA5ok_rt' or self.dataType == 'tp':
            self.filePath = './Data/WSDream/Dataset#1/'
        else:
            sys.stdout.write('\rData type error.')
            sys.exit()
        data = pd.read_csv(self.filePath + '%s_origin.txt' % self.dataType, sep='\t')
        return data, [data.iloc[:, 0].max() + 1, data.iloc[:, 1].max() + 1]

    def getTrainTest(self):
        train = pd.read_csv(self.filePath + '%s_train_%.2f.txt' % (self.dataType, self.density), sep='\t')
        test = pd.read_csv(self.filePath + '%s_test_%.2f.txt' % (self.dataType, self.density), sep='\t')
        sys.stdout.write("\rLoading completes.[%.2fs] userNum=%d | serviceNum=%d | density=%.2f | dataType=%s\n"
                         % (time() - self.start, self.shape[0], self.shape[1], self.density, self.dataType))
        return train, test

    def getTrainInstance(self, data):
        userID = np.array(data.iloc[:, 1])
        serviceID = np.array(data.iloc[:, 0])
        QoS = np.round(np.array(data.iloc[:, 2]), 3)
        u_q = np.array(data.iloc[:, [3, 4]])
        u_q2 = np.array(data.iloc[:,[12,13,14,15,16]] )
        s_q = np.array(data.iloc[:, [5,6]])
        s_q2 = np.array(data.iloc[:,[7,8,9,10,11]])

        return [userID, serviceID,u_q,u_q2,s_q,s_q2], QoS

    def getTestInstance(self, data):
        userID = np.array(data.iloc[:, 1])
        serviceID = np.array(data.iloc[:, 0])
        QoS = np.round(np.array(data.iloc[:, 2]), 3)
        u_q = np.array(data.iloc[:, [3, 4]])
        u_q2 = np.array(data.iloc[:,[12,13,14,15,16]] )


        s_q = np.array(data.iloc[:, [5,6]])
        s_q2 = np.array(data.iloc[:,[7,8,9,10,11]])



        return [userID, serviceID, u_q, u_q2, s_q, s_q2], QoS


