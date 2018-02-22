#coding:utf-8
import numpy as np
from graph import Graph


def generateDataSet():

    dataset = np.random.randn(10000,1000)
    #datalabel = np.random.randint(0,2,(10000,1))
    datalabel = np.random.rand(10000,1)
    #print dataset
    #print datalabel
    return dataset,datalabel


def test():
    dataset,datalabel = generateDataSet()

    graph = Graph(dataset,datalabel,'mse',0.01,opt='sgd')
    graph.addLayer(1000,512,'sigmoid')
    graph.addLayer(512,128,'sigmoid',l2=True)
    graph.addLayer(128,64,'sigmoid',keep_pro=0.5)
    graph.addLayer(64,16,'sigmoid')
    graph.addLayer(16,1,'sigmoid')
    for i in range(100):
        output =  graph.fowardOutput()
        cost = graph.computeCost(output)
        graph.backwardOutput(output)



if __name__=='__main__':
    test()


