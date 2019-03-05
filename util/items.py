import pickle
import re
import os
import numpy as np

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = (os.sep).join(dir_path.split(os.sep)[:-1])

itemInfo = {}
with open(dir_path+'/Data/itemInfo.pickle'.replace('/',os.sep),'rb') as f:
    itemInfo = pickle.load(f)

itemPrices = {}
with open(dir_path+'/Data/itemPrices.pickle'.replace('/',os.sep),'rb') as f:
    itemPrices = pickle.load(f)

itemCats = {}
with open(dir_path+'/Data/itemCats.pickle'.replace('/',os.sep),'rb') as f:
    itemCats = pickle.load(f)

viableItems = []
with open(dir_path+'/Data/viableItems.pickle'.replace('/',os.sep),'rb') as f:
    viableItems = pickle.load(f)

def getInfo(item):
    return itemInfo[item]

def getViableItems(

):
    return viableItems

def getPrices(item):
    return [a[1] for a in itemPrices[item]]

def getPriceChanges(item):
    p = getPrices(item)
    return [p[i+1]-p[i] for i in range(len(p)-1)]

def sma(prices, n=7):
    ret = np.cumsum(prices, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def ema(prices, n=14):
    s = 2/(n+1)
    arr = [prices[0]]
    for i in range(1,len(prices)):
        arr.append((prices[i]*s)+(arr[i-1]*(1-s)))
    return arr[n-1:]

def stochOscil(prices,n=14,k=3):
    kFast = []
    for i in range(n,len(prices)):
        sub = prices[i-n:i]
        l = min(sub)
        h = max(sub)
        d = h-l
        if d==0:
            d = 1
        kFast.append(((prices[i-1]-l)/d)*100)
    D = []
    for i in range(k,len(kFast)):
        D.append(np.mean(kFast[i-k:i]))
    return kFast,D

def momentum(prices,n=10):
    mom = []
    for i in range(n,len(prices)):
        mom.append(prices[i]-prices[i-n])
    momMA = []
    for i in range(n,len(mom)):
        momMA.append(np.mean(mom[i-n:i]))
    return mom, momMA

def macd(prices,nS=12,nL=26,nSign=9):
    s = ema(prices, n=nS)
    l = ema(prices, n=nL)
    m = []
    for i in range(-1*len(l),0):
        m.append(s[i]-l[i])
    s = 2 / (nSign + 1)
    arr = [m[0]]
    for i in range(1, len(m)):
        arr.append((m[i] * s) + (arr[i - 1] * (1 - s)))
    return m,arr

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
        if item1!=item2:
            m = min(sim)
            s = similarity(item1,item2)
            if s>m:
                idx = sim.index(m)
                sim[idx] = s
                items[idx] = item2
    return [(items[i],sim[i]) for i in range(n)]

def bollingerBands(data):
    means = []
    stds = []
    for i in range(20,len(data)):
        d = data[i-20:i]
        means.append(np.mean(d))
        stds.append(np.std(d))
    return np.array(means),np.array(stds)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    item = 'Abyssal_whip'

    p = getPrices(item)
    s3 = sma(item,n=3)
    s12 = sma(item,n=12)
    e= ema(item,n=10)
    print(len(p),len(s12),len(e))
    plt.plot(p[-1*len(s12):],label='Price')
    plt.plot(s3,label='SMA_3')
    plt.plot(s12,label='SMA_12')
    plt.plot(e,label='EMA')
    plt.legend()
    plt.xlim(0,90)
    plt.show()

    kFast , D = stochOscil(item)
    plt.plot(kFast,label='kFast')
    plt.plot(D,label='D')
    plt.legend()
    plt.show()

    m,mema = macd(item)
    plt.plot(m[-1*len(mema):],label='MACD')
    plt.plot(mema,label='Signal')
    plt.legend()
    plt.xlim(0,90)
    plt.show(m)

    mom,momMA = momentum(item)
    plt.plot(mom[-1*len(momMA):],label='Momentum')
    plt.plot(momMA,label='momMA')
    plt.legend()
    plt.xlim(0,90)
    plt.show(m)