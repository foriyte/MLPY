import numpy as np

from layer import FullConnectLayer
from softmax import softmaxLayer


class Graph():
    inputData = None
    labelData = None
    layerStack = []
    optMethod = None
    lossFunc = None
    batchsize = None
    alpha = None
    epsilon = 0.00000001
    opt = None

    def __init__(self,inputdata,labeldata,lossFunc,learnRate=0.01,opt='sgd'):
        self.inputData = inputdata
        if len(labeldata.shape) == 1:
            labeldata = labeldata.reshape([len(labeldata,1)])
        self.labelData = labeldata
        self.batchsize = len(inputdata)
        self.alpha = learnRate
        self.lossFunc = lossFunc
        self.opt = opt

    def addLayer(self,l,n,acfuncname,l1=False,l2=False,lamba=0.1,keep_pro=1.0,name=None):
        if len(self.layerStack) == 0:
            if l != self.inputData.shape[1]:
                raise BaseException('Exception: layer size is not match!')
        else:
            preLayer = self.layerStack[-1]
            if l!= preLayer.n:
                raise BaseException('Exception: layer size is not match!')
        if name is None:
            curLayer = FullConnectLayer(l,n,acfuncname,self.alpha,l1,l2,lamba,keep_pro,self.opt)
        elif name == 'softmax':
            maxclass = np.max(self.labelData)+1
            if n < maxclass:
                raise BaseException('Exception: class number is not match!')
            curLayer = softmaxLayer(l,n,self.alpha,l1,l2,lamba,self.opt)
            self.lossFunc = 'softmax'
        elif name == 'logistic':
            curLayer = FullConnectLayer(l,n,acfuncname,self.alpha,l1,l2,lamba,keep_pro,self.opt)
            self.lossFunc = 'log'
        self.layerStack.append(curLayer)

    def fowardOutput(self):
        temp = self.inputData
        for i in range(len(self.layerStack)):
            temp = self.layerStack[i].fowardOutput(temp)
        return temp

    def computeCost(self,outputData):
        if self.lossFunc == 'log':
            loss = -1*np.sum((np.dot(self.labelData.T,np.log(outputData+self.epsilon))+\
                    np.dot((1-self.labelData).T,np.log(1-outputData+self.epsilon))))/self.batchsize

        elif self.lossFunc == 'mse':
            loss = np.sum((outputData-self.labelData)**2)/self.batchsize

        elif self.lossFunc == 'softmax':
            loss = -np.sum(np.log(outputData[np.arange(len(outputData)),\
                    self.labelData.reshape(len(outputData))]+self.epsilon))/self.batchsize
            #for i in range(len(outputData)):
            #    loss = loss + -1* np.log(outputData[i][self.labelData[i][0]]+self.epsilon)
            #loss = loss/self.batchsize
        print 'loss:',loss
        return loss

    def backwardOutput(self,outputData):
        cur = None
        index = len(self.layerStack) -1
        if self.lossFunc == 'log':
            #outputData = outputData + self.epsilon
            da = (-1)*(self.labelData/(outputData+self.epsilon))+\
                    ((1-self.labelData)/(1-outputData+self.epsilon))
        elif self.lossFunc == 'mse':
            da = self.labelData - outputData
        elif self.lossFunc == 'softmax':
            da,df = self.layerStack[index].backwardOutput(self.labelData)
            self.layerStack[index].updateParameter(df)
            index = index -1
        #print 'pre: ',pre
        for i in range(len(self.layerStack)-1):
             temp = self.layerStack[index-i].backwardOutput(da)
             self.layerStack[index-i].updateParameter(da)
             da = temp

    def no_dropout(self):
        for layer in self.layerStack:
            layer.set_keep_pro(1)



