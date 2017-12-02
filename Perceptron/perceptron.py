#coding=utf-8

import numpy as np
import numpy.matlib
import random


class Perceptron():
    learnRate = 0.1
    iteration = 50
    iterationType = 'random'
    aomiga = None
    alpha = None

    def __init__(self,learnRate,iteration,iterationType):
        self.learnRate = learnRate
        self.iteration = iteration
        self.iterationType = iterationType

    def fit(self,data_X,data_Y):
        if type(data_X) != np.ndarray or type(data_Y) != np.ndarray:
            print 'Exception: data type is not match!'
            return

        if len(data_X.shape) != 2 or len(data_Y.shape) != 1:
            print 'Exception: data shape is not match!'
            return

        for p in data_Y:
            if p != 1 and p != -1:
                print 'Exception: label type only contains 1 or -1!'
                return


        num = data_X.shape[0]
        vnum = data_X.shape[1]+1
        self.alpha = np.zeros(num,dtype = np.float)
        errorlist = []

        #reprocess
        bvector = np.ones(num,dtype = np.int)
        bvector.shape = (num,1)
        MX = np.matrix(np.hstack((data_X,bvector)))
        TX = np.matlib.zeros((vnum,1),dtype = np.float)
        MY = np.matlib.zeros((1,num),dtype = np.float)


        for i in xrange(self.iteration):
            for j in xrange(num):
                TX += (MX[j]*self.alpha[j]*data_Y[j]).T
            MY = MX*TX
            for j in xrange(num):
                if MY[j][0]<=0:
                    errorlist.append(j)

            if errorlist:
                self.updateParameter(errorlist)
            else:
                self.aomiga = TX
                print 'training finished!'
                return
        self.aomiga = TX
        print 'training finished!'


    def updateParameter(self,errorlist):
        if self.iterationType == 'random':
            index = random.randint(0,len(errorlist)-1)
            index = errorlist[index]
            self.alpha[index] += self.learnRate

        elif self.iterationType == 'all':
            for i in errorlist:
                self.alpha[i] += self.learnRate

    def predict(self,data_X):

        if type(data_X) != np.ndarray:
            print 'Exception: data type is not match!'
            return
        if data_X.shape[1] != self.aomiga.shape[0]-1:
            print 'Exception: inputdata shape is not match!'
            return
        print self.aomiga
        num = data_X.shape[0]
        bvector = np.ones(num,dtype = np.int)
        bvector.shape = (num,1)

        MX = np.matrix(np.hstack((data_X,bvector)))

        result = MX*self.aomiga
        result = np.array(result,dtype = np.int).reshape(data_X.shape[0])
        for i in xrange(num):
            if result[i]<=0:
                result[i] = -1
            else:
                result[i] = 1
        return result











