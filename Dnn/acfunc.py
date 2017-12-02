import numpy as np

class Acfunc():
    name = None

    def __init__(self,name):
        self.name = name

    def computeOutput(self,z):
        e_z = np.exp(-1*z)
        if self.name == 'sigmoid':
            return 1/(1+e_z)
        elif self.name == 'tanh':
            ez = np.exp(z)
            return (ez-e_z)/(ez+e_z)
        elif self.name == 'relu':
            return np.maximum(0,z)

    def computeDiff(self,z):
        fz = self.computeOutput(z)
        if self.name == 'sigmoid':
            return fz*(1-fz)
        elif self.name == 'tanh':
            return 1-fz*fz
        elif self.name == 'relu':
            fz[fz>0] = 1
            return fz




