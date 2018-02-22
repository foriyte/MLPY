#coding:utf8
import numpy as np
from layer import *



class softmaxLayer(FullConnectLayer):

    df = None
    epsilon = 0.000000001
    def __init__(self,l,n,learnRate,l1=False,l2=False,lamda=0.1,opt='sgd'):
        super(softmaxLayer,self).__init__(l,n,None,learnRate=learnRate,l1=l1,l2=l2,\
                lamda=lamda,keep_pro=1.0,opt=opt)
        self.beta = 0

    def fowardOutput(self,inputData):
        self.z = super(softmaxLayer,self).fowardOutput(inputData)
        f_max = np.max(self.z)
        pro_base = np.sum(np.exp(self.z-f_max),axis=1).reshape([len(self.z),1])
        self.gz = np.exp(self.z-f_max)/(pro_base+self.epsilon)
        return self.gz

    def backwardOutput(self,inputA):
        """
        对每个样本计算其loss,loss为对应label列的softmax概率的负对数。
        然后对exp(wix)求梯度，分为两种情况，wi对应的label为正确分类或不是正确分类。
        两种情况得到不同的梯度,对应为正确label的梯度为pi－1，不正确为pi，pi为对应i 类的概率。
        最后对omega的每一列求梯度时，汇集所有样本，有的样本为情况1，有的为情况2。
        """
        keeppro = np.zeros_like(self.gz)
        label = inputA.reshape(len(inputA))
        keeppro[np.arange(len(label)),label] = 1.0
        self.df = self.gz-keeppro
        self.df =self.df/len(inputA)
        return np.dot(self.df,self.omega.T),self.df

