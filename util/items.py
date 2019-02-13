import pickle
import re
import os
import numpy as np

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/'.join(dir_path.split('/')[:-1])

itemInfo = {}
with open(dir_path+'/Data/itemInfo.pickle','rb') as f:
    itemInfo = pickle.load(f)

itemPrices = {}
with open(dir_path+'/Data/itemPrices.pickle','rb') as f:
    itemPrices = pickle.load(f)

itemCats = {}
with open(dir_path+'/Data/itemCats.pickle','rb') as f:
    itemCats = pickle.load(f)

viableItems = []
with open(dir_path+'/Data/viableItems.pickle','rb') as f:
    viableItems = pickle.load(f)

def getInfo(item):
    return itemInfo[item]

def getPrices(item):
    return [a[1] for a in itemPrices[item]]

def getPriceChanges(item):
    p = getPrices(item)
    return [p[i+1]-p[i] for i in range(len(p)-1)]

def movingAverage(item, n=7) :
    ret = np.cumsum(getPrices(item), dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def getDates(item):
    return [a[0] for a in itemPrices[item]]

def getQuants(item):
    return [a[2] for a in itemPrices[item] if a[2] > 0]

def getCats(item):
    return itemCats[item]

def similarity(item1,item2):
    name1 = [a for a in re.split('[^A-Za-z]',item1) if len(a)>2]
    name2 = [a for a in re.split('[^A-Za-z]',item2) if len(a)>2]
    n = len(set(name1)&set(name2))
    p = len(set(itemCats[item1])&set(itemCats[item2]))
    return n*5+p

def getSimilarItems(item1,n=5):
    sim = [0]*n
    items = [None]*n
    for item2 in itemCats:
        m = min(sim)
        s = similarity(item1,item2)
        if s>m:
            idx = sim.index(m)
            sim[idx] = s
            items[idx] = item2
    return [(items[i],sim[i]) for i in range(n)]
