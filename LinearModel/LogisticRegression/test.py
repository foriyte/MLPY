#coding:utf-8
import numpy as np
from LogisticRegression import LogisticRegression


def generateDataSet():

    dataset = np.random.randn(1000,100)
    datalabel = np.random.randint(0,2,(1000,1))
    #print dataset
    #print datalabel
    return dataset,datalabel


def test():
    dataset,datalabel = generateDataSet()

    logis = LogisticRegression()
    logis.fit(dataset,datalabel)
    pre = logis.predict(dataset)
    print pre


if __name__=='__main__':
    test()


