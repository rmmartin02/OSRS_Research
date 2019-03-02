import matplotlib.pyplot as plt
import numpy as np
import pickle

'''
import keras
from keras.layers import InputLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

from sklearn.preprocessing import MinMaxScaler
'''
import util.items as items
import util.trading_systems as ts

from pandas_datareader import data
import pandas as pd

if __name__ == "__main__":

    item = "Lassar_teleport-"

    p = items.getPrices(item)
    s3 = items.sma(p, n=3)
    s12 = items.sma(p, n=12)
    e = items.ema(p, n=10)
    plt.plot(p[-1 * len(s12):], label='Price')
    plt.plot(s3, label='SMA_3')
    plt.plot(s12, label='SMA_12')
    plt.plot(e, label='EMA')
    plt.legend()
    plt.xlim(0, 90)
    #plt.show()

    kFast, D = items.stochOscil(p)
    plt.plot(kFast, label='kFast')
    plt.plot(D, label='D')
    plt.legend()
    #plt.show()

    m, mema = items.macd(p)
    plt.plot(m[-1 * len(mema):], label='MACD')
    plt.plot(mema, label='Signal')
    plt.legend()
    plt.xlim(0, 90)
    #plt.show(m)

    mom, momMA = items.momentum(p)
    plt.plot(mom[-1 * len(momMA):], label='Momentum')
    plt.plot(momMA, label='momMA')
    plt.legend()
    plt.xlim(0, 90)
    plt.show(m)

    bl = int(items.getInfo(item)['buyLimit'])
    budget = bl * max(p) + 1
    
    stcOscProf = ts.crossOverProfit(kFast,D,p,bl,budget)
    macdProf = ts.crossOverProfit(m, mema, p, bl, budget)
    momProf = ts.crossOverProfit(mom, momMA, p, bl, budget)
    persProf = ts.persistanceProfit(p,bl,budget)
    perfProf = ts.perfectProfit(p,bl,budget)
    print(stcOscProf[-1], macdProf[-1], momProf[-1],persProf[-1],perfProf[-1])
    plt.plot(stcOscProf)
    plt.plot(macdProf)
    plt.plot(momProf)
    plt.plot(persProf)
    #plt.plot(perfProf)
    plt.show()

    stcOscProf = ts.crossOverProfit(kFast[-120:],D[-120:],p[-120:],bl,budget)
    macdProf = ts.crossOverProfit(m[-120:], mema[-120:], p[-120:], bl, budget)
    momProf = ts.crossOverProfit(mom[-120:], momMA[-120:], p[-120:], bl, budget)
    persProf = ts.persistanceProfit(p[-120:],bl,budget)
    perfProf = ts.perfectProfit(p[-120:],bl,budget)
    print(stcOscProf[-1], macdProf[-1], momProf[-1],persProf[-1],perfProf[-1])
    plt.plot(stcOscProf)
    plt.plot(macdProf)
    plt.plot(momProf)
    plt.plot(persProf)
    #plt.plot(perfProf)
    plt.show()