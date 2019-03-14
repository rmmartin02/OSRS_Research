import sys
import os
address = (os.sep).join(os.getcwd().split(os.sep)[:-2])
print(address)
sys.path.append(address)
print(sys.path)
import util.items as items
import util.trading_systems as ts
import util.regression_model as rm
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    item = "Yew_logs"

    prices = items.getPrices(item)
    changes = items.getPriceChanges(item)
    sma3 = items.sma(changes,3)
    ema3 = items.ema(changes,3)

    featSizes = [10]
    model = rm.RegressionModel(sma3,[changes],featSizes,'sigmoid',sum(featSizes),sum(featSizes),.8,.9)
    model.train(30,8)
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
    best = [profit[-1],buySig,sellSig,len([a for a in buySigs if a==True]),len([a for a in sellSigs if a==True])]
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

    plt.plot(perf, label = 'Perfect')
    plt.plot(pers,label = 'Persist')
    plt.plot(BaH, label = 'B&H')
    plt.plot(profit, label = 'Model')
    plt.legend()
    plt.show()