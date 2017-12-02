#coding:utf-8
import numpy as np
from acfunc import Acfunc


class FullConnectLayer():
    l = None
    n = None
    omega = None
    beta = None
    alpha = None
    acfunc = None
    z = None
    gz = None
    inputa = None

    #regularzition
    l1 = None
    l2 = None
    lamda = None
    keep_pro = None
    dropout = None

    def __init__(self,l,n,acfuncName,learnRate=0.01,l1=False,l2=False,lamda=0.1,keep_pro=1.0):
        self.l = l
        self.n = n
        self.acfunc = Acfunc(acfuncName)
        if self.omega == None or self.beta == None:
            self.omega = np.random.randn(self.l,self.n)
            self.beta = np.zeros([1,self.n])
        self.alpha = learnRate

        self.l1 = l1
        self.l2 = l2
        self.lamda = lamda
        self.keep_pro = keep_pro

    #正向传播
    def fowardOutput(self,inputData):
        if type(inputData) != np.ndarray:
            print 'Exception: data type is not match!'
            return
        if inputData.shape[1] != self.l:
            print 'Exception: data shape is not match!'
            return
        self.inputa = inputData
        if self.keep_pro<1:
            self.dropout = np.random.rand(inputData.shape[0],inputData.shape[1])<self.keep_pro
            inputData = np.multiply(inputData,self.dropout)
            #这样在测试中,就需要使用权重比例推断
            inputData = inputData/self.keep_pro
        self.z = np.dot(inputData,self.omega) + self.beta
        self.gz = self.acfunc.computeOutput(self.z)
        return self.gz

    #反向传播
    def backwardOutput(self,inputA):
        return np.dot(inputA*self.acfunc.computeDiff(self.z),self.omega.T)

    def updateParameter(self,da):
        dz = da*self.acfunc.computeDiff(self.z)/len(self.inputa)
        dw = np.dot(self.inputa.T,dz)
        if self.l2 == True:
            dw = dw + (self.lamda/len(self.inputa))*self.omega
        db = np.sum(dz,axis=0,keepdims=1)
        self.omega = self.omega - self.alpha*dw
        self.beta = self.beta - self.alpha*db

    def set_keep_pro(self,keep_pro):
        self.keep_pro = keep_pro










