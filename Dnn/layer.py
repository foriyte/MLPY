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

    def __init__(self,l,n,acfuncName,learnRate=0.01):
        self.l = l
        self.n = n
        self.acfunc = Acfunc(acfuncName)
        if self.omega == None or self.beta == None:
            self.omega = np.random.randn(self.l,self.n)
            self.beta = np.zeros([1,self.n])
        self.alpha = learnRate

    def fowardOutput(self,inputData):
        if type(inputData) != np.ndarray:
            print 'Exception: data type is not match!'
            return
        if inputData.shape[1] != self.l:
            print 'Exception: data shape is not match!'
            return
        self.inputa = inputData
        self.z = np.dot(inputData,self.omega) + self.beta
        self.gz = self.acfunc.computeOutput(self.z)
        return self.gz

    def backwardOutput(self,inputA):
        h = self.acfunc.computeDiff(self.z)
        #print self.omega.shape
        #print h.shape
        hi = self.acfunc.computeDiff(self.z)
        ha = inputA*hi
        return np.dot(inputA*self.acfunc.computeDiff(self.z),self.omega.T)

    def updateParameter(self,da):
        dz = da*self.acfunc.computeDiff(self.z)/len(self.inputa)
        dw = np.dot(self.inputa.T,dz)
        db = np.sum(dz,axis=0,keepdims=1)
        self.omega = self.omega - self.alpha*dw
        self.beta = self.beta - self.alpha*db


















