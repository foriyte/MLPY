import numpy as np

class LogisticRegression():
    learnRate = None
    iteration = None
    iterationType = None
    aomiga = None
    beta = None
    m = None
    n = None


    def __init__(self,learnRate = 0.01,iteration = 50,iterationType = 'random'):
       self.learnRate = learnRate
       self.iteration = iteration
       self.iterationType = iterationType

    def fit(self,data_X,data_Y):
        if type(data_X) != np.ndarray or type(data_Y) != np.ndarray:
            print 'Exception: data type is not match!'
            return
        if  len(data_X.shape) != 2 or len(data_Y.shape) != 2:
            print 'Exception: data shape is not match!'
            return
        self.m = data_X.shape[0]
        self.n = data_X.shape[1]
        self.aomiga = np.random.randn(self.n,1)
        self.beta = 0

        for i in range(self.iteration):
            loss = self.sgd(data_X,data_Y)


    def sgd(self,data_X,data_Y):
        data_Z = np.dot(data_X,self.aomiga)+self.beta
        y_hat = 1/(1+np.exp(data_Z*-1))
        loss = -1*(np.dot(data_Y.T,np.log(y_hat))+np.dot((1-data_Y).T,np.log(1-y_hat)))/self.m

        dw = -1*(np.dot(data_X.T,(data_Y-y_hat)))/self.m
        db = -1*np.sum(data_Y-y_hat)/self.m

        self.aomiga = self.aomiga - self.learnRate*dw
        self.beta = self.beta - self.learnRate*db

        return loss


    def predict(self,data_X,threshold = 0.5):
        if self.aomiga == None or self.beta == None or self.m == None or self.n == None:
            print 'there is not model!'
            return

        if type(data_X) != np.ndarray:
            print 'Exception: data type is not match!'
            return
        if data_X.shape[1] != self.aomiga.shape[0]:
            print 'Exception: data shape is not match'
            return

        data_Z = np.dot(data_X,self.aomiga)+self.beta
        y_hat = 1/(1+np.exp(data_Z*-1))

        result = np.zeros([self.m,1])
        for i in range(len(y_hat)):
            if y_hat[i][0]>threshold:
                result[i][0] = 1
        return y_hat,result

