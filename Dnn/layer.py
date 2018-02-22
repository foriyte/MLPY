#coding:utf-8
import numpy as np
from acfunc import Acfunc


class FullConnectLayer(object):
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

    opt = None
    #动量
    Momentum_v = None
    Momentum_alpha = None
    Momentum_epsilon = None

    #RMSProp
    RMSProp_r = None
    RMSProp_rho = None
    RMSProp_xi = None
    RMSProp_epsilon = None

    #Adam
    Adam_rho1 = None
    Adam_rho2 = None
    Adam_epsilon = None
    Adam_xi = None
    Adam_s = None #1阶距
    Adam_r = None #2阶距
    Adam_t = None


    def __init__(self,l,n,acfuncName,learnRate=0.01,\
            l1=False,l2=False,lamda=0.1,keep_pro=1.0,opt='sgd'):
        self.l = l
        self.n = n
        if acfuncName is not None:
            self.acfunc = Acfunc(acfuncName)
        if self.omega is  None or self.beta is None:
            self.omega = np.random.randn(self.l,self.n)
            self.beta = np.zeros([1,self.n])
        self.alpha = learnRate

        self.l1 = l1
        self.l2 = l2
        self.lamda = lamda
        self.keep_pro = keep_pro

        self.opt = opt

        #动量参数初始化
        self.Momentum_v = 0.0
        self.Momentum_alpha = 0.5
        self.Momentum_epsilon = learnRate

        #RMSPro参数初始化
        self.RMSProp_xi = 0.000001
        self.RMSProp_epsilon = learnRate
        self.RMSProp_r = 0.0
        self.RMSProp_rho = 0.9

        #Adam参数初始化
        self.Adam_s = 0.0
        self.Adam_r = 0.0
        self.Adam_t = 1
        self.Adam_xi = 0.00000001
        self.Adam_rho1 = 0.9
        self.Adam_rho2 = 0.9
        self.Adam_epsilon = 0.001


    #正向传播
    def fowardOutput(self,inputData):
        if type(inputData) != np.ndarray:
            raise BaseException('Exception: data type is not match!')
        if inputData.shape[1] != self.l:
            raise BaseException('Exception: data shape is not match!')
        self.inputa = inputData
        if self.keep_pro<1:
            self.dropout = np.random.rand(inputData.shape[0],inputData.shape[1])<self.keep_pro
            inputData = np.multiply(inputData,self.dropout)
            #这样在测试中,就需要使用权重比例推断
            inputData = inputData/self.keep_pro
        self.z = np.dot(inputData,self.omega) + self.beta
        if self.acfunc is not None:
            self.gz = self.acfunc.computeOutput(self.z)
        else:
            self.gz = self.z
        return self.gz

    #反向传播
    def backwardOutput(self,inputA):
        return np.dot(inputA*self.acfunc.computeDiff(self.z),self.omega.T)

    def updateParameter(self,da):
        if self.acfunc is not None:
            dz = da*self.acfunc.computeDiff(self.z)/len(self.inputa)
        else:
            dz = da
        dw = np.dot(self.inputa.T,dz)
        if self.l2 == True:
            dw = dw + (self.lamda/len(self.inputa))*self.omega
        db = np.sum(dz,axis=0,keepdims=1)

        if self.opt == 'sgd':
            dw = -1*self.alpha*dw

        elif self.opt == 'Momentum':
            dw = self.Momentum_alpha*self.Momentum_v - self.alpha*dw
            self.Momentum_v = dw

        elif self.opt == 'RMSProp':
            self.RMSProp_r = self.RMSProp_rho*self.RMSProp_r + (1-self.RMSProp_rho)*dw*dw
            dw = -1*(self.RMSProp_epsilon/np.sqrt(self.RMSProp_xi+self.RMSProp_r))*dw

        elif self.opt == 'Adam':
            self.Adam_s = self.Adam_rho1*self.Adam_s+(1-self.Adam_rho1)*dw
            self.Adam_r = self.Adam_rho2*self.Adam_r+(1-self.Adam_rho2)*dw*dw
            Adam_s_hat = self.Adam_s/(1-np.power(self.Adam_rho1,self.Adam_t))
            Adam_r_hat = self.Adam_r/(1-np.power(self.Adam_rho2,self.Adam_t))
            dw = (-1*self.Adam_epsilon*Adam_s_hat)/(np.sqrt(Adam_r_hat)+self.Adam_xi)
            self.Adam_t = self.Adam_t + 1

        #update
        self.omega = self.omega + dw
        self.beta = self.beta - self.alpha*db

    def set_keep_pro(self,keep_pro):
        self.keep_pro = keep_pro





