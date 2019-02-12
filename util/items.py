import pickle
import re
import os

itemInfo = {}
with open(os.curdir+'/Data/itemInfo.pickle','rb') as f:
    itemInfo = pickle.load(f)

itemPrices = {}
with open(os.curdir+'/Data/itemPrices.pickle','rb') as f:
    itemPrices = pickle.load(f)

itemCats = {}
with open(os.curdir+'/Data/itemCats.pickle','rb') as f:
    itemCats = pickle.load(f)

viableItems = []
with open(os.curdir+'/Data/viableItems.pickle','rb') as f:
    viableItems = pickle.load(f)

def getItemInfo(item):
    return itemInfo[item]

def getItemPrices(item):
    return [a[1] for a in itemPrices[item]]

def getPriceChanges(item):
    p = getItemPrices(item)
    return [p[i+1]-p[i] for i in range(len(p)-1)]

def getItemDates(item):
    return [a[0] for a in itemPrices[item]]

def getItemQuants(item):
    return [a[2] for a in itemPrices[item] if a[2] > 0]

def getItemCats(item):
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
