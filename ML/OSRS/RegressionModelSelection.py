import sys
import os
address = (os.sep).join(os.getcwd().split(os.sep)[:-2])
print(address)
sys.path.append(address)
print(sys.path)
import util.items as items
import util.trading_systems as ts
import util.regression_model as rm
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from datetime import datetime


if __name__ == "__main__":

    item = "Abyssal_whip"

    with open("../../Data/top100Items.txt","r") as f:
        itemIndex = f.readlines()
    itemIndex = [
        "Yew_logs",
        "Maple_logs",
        "Willow_logs",
        "Oak_logs",
        "Magic_logs",
    ]
    pri = [items.getPrices(item) for item in itemIndex]
    minLength = len(pri[0])
    for p in pri:
        if len(p)<minLength:
            minLength = len(p)
    prices = [0] * minLength
    for i in range(-1,-1*minLength,-1):
        s = 0
        # should probably scale the prices here for doing indexes?
        for a in pri:
            s+=a[i]
        s/=len(pri)
        prices[i] = s


    start = datetime(2015, 3, 1)

    end = datetime(2019, 3, 1)

    f = web.DataReader('AAPL', 'iex', start, end)
    prices = np.array(f["close"])

    #prices = items.getPrices(item)
    changes = items.getPriceChanges(prices)


    sma3 = items.sma(changes,3)
    ema3 = items.ema(changes,3)

    featSizes = [21]
    model = rm.RegressionModel(changes,[changes],featSizes,'sigmoid',sum(featSizes),sum(featSizes),.8,.9)
    model.train(50,16)
    model.graphLoss()
    model.graphMAE()
    model.graphPredict()


    bl = int(items.getInfo(item)['buyLimit'])
    test_prices = prices[-1 * len(model.y_test) + -1*len(model.y_val):-1 * len(model.y_test)]
    budget = test_prices[0]
    y_pred = model.predict(model.x_val)

    buySigs = [a >= 0 for a in y_pred]
    sellSigs = [a <= 0 for a in y_pred]
    profit = ts.modelProfit(buySigs, sellSigs, test_prices, bl, budget)[-1]
    best = [profit,0,0,len([a for a in buySigs if a==True]),len([a for a in sellSigs if a==True])]
    print(best)

    for buySig in np.linspace(-1,1,20):
        for sellSig in np.linspace(1,-1,20):
            buySigs = [a >= buySig for a in y_pred]
            sellSigs = [a <= sellSig for a in y_pred]
            profit = ts.modelProfit(buySigs,sellSigs,test_prices,bl,budget)[-1]
            if profit>best[0]:
                best = [profit,buySig,sellSig,len([a for a in buySigs if a==True]),len([a for a in sellSigs if a==True])]


    test_prices = prices[-1 * len(model.y_test):]
    budget = test_prices[0]
    y_pred = model.predict(model.x_test)

    buySigs = [a >= best[1] for a in y_pred]
    sellSigs = [a <= best[2] for a in y_pred]
    profit = ts.modelProfit(buySigs, sellSigs, test_prices, bl, budget)
    best = [profit[-1],best[1],best[2],len([a for a in buySigs if a==True]),len([a for a in sellSigs if a==True])]
    print(best)

    plt.plot(test_prices)
    top = max(test_prices)
    bot = max(test_prices)
    for i in range(len(test_prices)):
        if buySigs[i]:
            plt.axvline(i,color='g')
        if sellSigs[i]:
            plt.axvline(i,color='r')
    plt.show()

    perf,pers,BaH = ts.baselines(test_prices,bl,budget)
    print(perf[-1],pers[-1],BaH[-1])
    print(pers)

    plt.plot(perf, label = 'Perfect')
    plt.plot(pers,label = 'Persist')
    plt.plot(BaH, label = 'B&H')
    plt.plot(profit, label = 'Model')
    plt.legend()
    plt.show()