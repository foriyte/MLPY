
import numpy as np


class LinearRegression:
    aomiga = None
    regressType = 'lsm'
    iterationNum = 50
    learnRate = 0.1


    def __init__(self,regressType,iterationNum,learnRate):
        self.regressType = regressType
        self.iterationNum = iterationNum
        selt.learnRate = learnRate

    def fit(self,data_X,data_Y):
        
