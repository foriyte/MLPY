#coding=utf-8

import perceptron
import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    gzj = perceptron.Perceptron(0.1,10,'all')

    df = pd.read_csv('test.csv')
    num = len(df)
    a = np.array(df.x1).reshape(num,1)
    b = np.array(df.x2).reshape(num,1)
    data_x = np.hstack((a,b))

    data_y = np.array(df.y)

    gzj.fit(data_x,data_y)
    test = np.array([[25,25]])
    print gzj.predict(test)
