#coding:utf-8
import numpy as np

class Acfunc():
    name = None
    epsilon = 0.00000001

    def __init__(self,name):
        self.name = name

    def computeOutput(self,z):
        #防止溢出,但是影响性能
        #z[z<-709] = -709.0
        e_z = np.exp(-1*z)
        if self.name == 'relu':
            return np.maximum(0,z)
        elif self.name == 'sigmoid':
            return 1/(1+e_z)
        elif self.name == 'tanh':
            ez = np.exp(z)
            return (ez-e_z)/(ez+e_z)

    def computeDiff(self,z):
        fz = self.computeOutput(z)
        if self.name == 'sigmoid':
            return fz*(1-fz)
        elif self.name == 'tanh':
            return 1-fz*fz
        elif self.name == 'relu':
            fz[fz>0] = 1
            return fz




