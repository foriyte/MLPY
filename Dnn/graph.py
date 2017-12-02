import numpy as np

from layer import FullConnectLayer


class Graph():
    inputData = None
    labelData = None
    layerStack = []
    optMethod = None
    lossFunc = None
    batchsize = None
    alpha = None
    epsilon = 0.00000001

    def __init__(self,inputdata,labeldata,lossFunc,learnRate=0.01):
        self.inputData = inputdata
        self.labelData = labeldata
        self.batchsize = len(inputdata)
        self.alpha = learnRate
        self.lossFunc = lossFunc

    def addLayer(self,l,n,acfuncname,l1=False,l2=False,lamba=0.1,keep_pro=1.0):
        if len(self.layerStack) == 0:
            if l != self.inputData.shape[1]:
                print 'Exception: layer size is not match!'
                return
            curLayer = FullConnectLayer(l,n,acfuncname,self.alpha,l1,l2,lamba,keep_pro)
            self.layerStack.append(curLayer)
        else:
            preLayer = self.layerStack[-1]
            if l!= preLayer.n:
                print 'Exception: layer size is not match!'
                return
            curLayer = FullConnectLayer(l,n,acfuncname,self.alpha,l1,l2,lamba,keep_pro)
            self.layerStack.append(curLayer)

    def fowardOutput(self):
        temp = self.inputData
        for i in range(len(self.layerStack)):
            temp = self.layerStack[i].fowardOutput(temp)
        return temp

    def computeCost(self,outputData):
        if self.lossFunc == 'log':
            loss = -1*(np.dot(self.labelData.T,np.log(outputData+self.epsilon))+\
                    np.dot((1-self.labelData).T,np.log(1-outputData+self.epsilon)))/self.batchsize
            print 'loss:',loss
            return loss
        elif self.lossFunc == 'mse':
            loss = np.sum((outputData-self.labelData)**2)/self.batchsize
            print 'loss:',loss
            return loss

    def backwardOutput(self,outputData):
        pre = None
        if self.lossFunc == 'log':
            #outputData = outputData + self.epsilon
            da = (-1)*(self.labelData/(outputData+self.epsilon))+\
                    ((1-self.labelData)/(1-outputData+self.epsilon))
        cur = None
        index = len(self.layerStack) -1
        #print 'pre: ',pre
        for i in range(len(self.layerStack)):
             self.layerStack[index-i].updateParameter(da)
             da = self.layerStack[index-i].backwardOutput(da)

    def no_dropout(self):
        for layer in self.layerStack:
            layer.set_keep_pro(1)



