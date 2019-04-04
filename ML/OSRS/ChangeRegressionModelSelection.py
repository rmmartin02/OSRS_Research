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

    f = web.DataReader('F', 'iex', start, end)
    prices = np.array(f["close"])

    prices = items.getPrices(item)
    changes = items.getPercentChanges(prices)

    sma5 = items.sma(changes,5)
    ema5 = items.ema(changes,5)

    #[5.13916970484437e-05, 0.004441426996744403]
    #[0.00012173630560057181, 0.00888896041819422]

    featSizes = [21]
    print(int(sum(featSizes)**.5))
    model = rm.RegressionModel(changes,[changes],featSizes,'sigmoid',sum(featSizes),sum(featSizes),.8,.9)

    if False:
        similar = items.getSimilarItems(item,3)
        simChanges = changes
        for s in similar:
            print(s)
            simChanges = simChanges + items.getPriceChanges(items.getPrices(s[0]))
        model.changeTrainingData(simChanges,[simChanges],[21])

    model.train(50,16)

    prices = items.getPrices(item)
    changes = items.getPercentChanges(prices)

    model.graphLoss()
    model.graphMAE()
    model.graphPredict()
    print(model.getScore())

    print("prices lengths same", len(prices), len(model.y_train) + len(model.y_test) + len(model.y_val))

    # try to optimize out small predictions
    test_prices = prices[len(model.y_train):len(model.y_train) + len(model.y_val)]
    budget = test_prices[0] * 101 - 1
    y_pred = model.predict(model.x_val)

    print(len(test_prices), len(y_pred))

    best = [-10000, 0, 0]
    for buySig in np.linspace(-1, 1, 20):
        for sellSig in np.linspace(1, -1, 20):
            buySigs = [y_pred[i] >= buySig for i in
                       range(1, len(y_pred))]
            buySigs = [False] + buySigs
            sellSigs = [y_pred[i] <= sellSig for i in
                        range(1, len(y_pred))]
            sellSigs = sellSigs + [False]
            profit = ts.modelProfit(buySigs, sellSigs, test_prices, budget)[-1]
            if profit > best[0]:
                best = [profit, buySig, sellSig]

    test_prices = prices[-1 * len(model.y_test):]
    budget = test_prices[0] * 101 - 1
    y_pred = model.predict(model.x_test)

    print(len(test_prices), len(y_pred))

    buySigs = [test_prices[i + 1] >= test_prices[i] for i in range(0, len(test_prices) - 1)]
    buySigs = buySigs + [False]
    sellSigs = [test_prices[i + 1] <= test_prices[i] for i in range(0, len(test_prices) - 1)]
    sellSigs = sellSigs + [False]
    print("lengths", len(buySigs), len(sellSigs), len(test_prices))
    perf = ts.modelProfit(buySigs, sellSigs, test_prices, budget)

    buySigs = [test_prices[i] >= test_prices[i - 1] for i in range(1, len(test_prices))]
    buySigs = [False] + buySigs
    sellSigs = [test_prices[i] <= test_prices[i - 1] for i in range(1, len(test_prices))]
    sellSigs = [False] + sellSigs
    print("lengths", len(buySigs), len(sellSigs), len(test_prices))
    pers = ts.modelProfit(buySigs, sellSigs, test_prices, budget)

    BaH = [(test_prices[i] / test_prices[0]) - 1 for i in range(len(test_prices))]

    smaProf = ts.crossOverProfit(items.sma(test_prices, 3), items.sma(test_prices, 12), test_prices,
                                 budget)
    stchOsc = items.stochOscil(test_prices, 3, 5)
    stchOscProf = ts.crossOverProfit(stchOsc[0], stchOsc[1], test_prices, budget)
    mom = items.momentum(test_prices, 10)
    momProf = ts.crossOverProfit(mom[0], mom[1], test_prices, budget)

    buySigs = [y_pred[i] >= 0 for i in range(1, len(y_pred))]
    buySigs = [False] + buySigs
    sellSigs = [y_pred[i] < 0 for i in range(1, len(y_pred))]
    sellSigs = [False] + sellSigs
    print("lengths", len(buySigs), len(sellSigs), len(test_prices))
    profit = ts.modelProfit(buySigs, sellSigs, test_prices, budget)

    buySigs = [y_pred[i] >= best[1] for i in range(1, len(y_pred))]
    buySigs = [False] + buySigs
    sellSigs = [y_pred[i] - y_pred[i - 1] <= best[2] for i in range(1, len(y_pred))]
    sellSigs = sellSigs + [False]
    profit_opt = ts.modelProfit(buySigs, sellSigs, test_prices, budget)

    smaProf_Pred = ts.crossOverProfit(items.sma(y_pred, 3), items.sma(y_pred, 12), test_prices, budget)
    stchOsc = items.stochOscil(y_pred, 3, 5)
    stchOscProf_Pred = ts.crossOverProfit(stchOsc[0], stchOsc[1], test_prices, budget)
    mom = items.momentum(y_pred, 10)
    momProf_Pred = ts.crossOverProfit(mom[0], mom[1], test_prices, budget)

    print(perf[-1], pers[-1], BaH[-1])
    print(profit[-1], profit_opt[-1], best[1], best[2])
    print(smaProf[-1], stchOscProf[-1], momProf[-1])
    print(smaProf_Pred[-1], stchOscProf_Pred[-1], momProf_Pred[-1])