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

    #get list of items (probably divide this up)
    with open("../../Data/top100Items.txt","r") as f:
        itemIndex = f.readlines()

    #go through list training model for each item
    toWrite = []

    item = "Abyssal_whip"
    toWrite.append(item)

    prices = items.getPrices(item)
    changes = items.getPriceChanges(prices)
    toWrite.append(len(prices))

    featSizes = [21]
    model = rm.RegressionModel(changes,[changes],featSizes,'sigmoid',sum(featSizes),sum(featSizes),.8,.9)

    beforeScore = model.getScore()
    toWrite.append(beforeScore[0])
    toWrite.append(beforeScore[1])

    model.train(50,16)

    print(model.getScore())
    afterScore = model.getScore()
    toWrite.append(afterScore[0])
    toWrite.append(afterScore[1])


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

    toWrite.append(bl)
    toWrite.append(budget)
    toWrite.append(test_prices)

    buySigs = [a >= best[1] for a in y_pred]
    sellSigs = [a <= best[2] for a in y_pred]
    profit = ts.modelProfit(buySigs, sellSigs, test_prices, bl, budget)
    best = [profit[-1],best[1],best[2],len([a for a in buySigs if a==True]),len([a for a in sellSigs if a==True])]


    toWrite.append(best[1])
    toWrite.append(best[2])



    perf,pers,BaH = ts.baselines(test_prices,bl,budget)
    print(profit[-1],perf[-1],pers[-1],BaH[-1])
