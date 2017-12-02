#coding:utf-8


import numpy as np
import random

localname = 'data.txt'

def readData():
    rid = []
    weight = []
    cyc = []
    origweight = []
    with open(localname,'r') as rf:
        for line in rf.readlines():
            blocks = line.strip().split('\t')
            rid.append(blocks[0])
            weight.append(float(blocks[1]))
            cyc.append(float(blocks[2]))
            origweight.append(float(blocks[3]))
            if weight[-1]<origweight[-1]:
                weight[-1] = weight[-1] + origweight[-1]/cyc[-1]
    return rid,weight,cyc,origweight

def randomAlgo(weight):
    allweight = sum(weight)
    relaweight = []
    pre = 0.0
    for i in weight:
        pre = pre + i/allweight
        relaweight.append(pre)
    value = random.randint(0,1000)
    value = 1.0*value/1000
    level = -1
    for i in xrange(len(relaweight)-1):
        if value>relaweight[i] and value<relaweight[i+1]:
            level = i
            break
    weight[level] = 0.0
    return level,weight

def randomSelect(rid,weight,cyc,origweight,num):
    res = []
    for i in range(num):
        level,weight = randomAlgo(weight)
        res.append(rid[level])
    writeDate(rid,weight,cyc,origweight)
    return res

def writeDate(rid,weight,cyc,origweight):
    with open(localname,'wb') as wf:
        for i in xrange(len(rid)):
            wf.write(rid[i]+'\t'+str(weight[i])+'\t'+str(cyc[i])+'\t'+str(origweight[i])+'\n')

if __name__=='__main__':
    a,b,c,d = readData()
    res = randomSelect(a,b,c,d,3)
    print res
