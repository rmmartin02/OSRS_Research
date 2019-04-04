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
from sklearn.preprocessing import StandardScaler

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

    def scale(p):
        m = max(p)
        arr = [0.0] * len(p)
        for i in range(len(p)):
            arr[i] = p[i]/m
        return arr

    prices = items.getPrices(item)
    plt.plot(prices)
    plt.show()

    sma5 = items.sma(prices,5)
    ema5 = items.ema(prices,5)

    #all [0.6288924502570609, 0.44649100230618133]
    #no ema [0.5967839539733621, 0.4365778479644721]
    #no sma [0.6045633982520005, 0.44838545833238597]
    #no changes [0.605660682296777, 0.4360905327438879]

    #all prayer pots [0.6392273480790555, 0.5626463457643985]
    #only one [0.6460398975707358, 0.5779683970808983]

    featSizes = [10,5,5]
    print(int(sum(featSizes)**.5))
    print('lengths',len(prices[2:]),len(prices[:-2]))
    model = rm.RegressionModel(sma5[2:],[prices[:-2],sma5[:-2],ema5[:-2]],featSizes,'sigmoid',int(sum(featSizes)**.5),0,.8,.9)

    if False:
        similar = items.getSimilarItems(item,3)
        simChanges = changes
        for s in similar:
            print(s)
            simChanges = simChanges + items.getPriceChanges(items.getPrices(s[0]))
        model.changeTrainingData(simChanges,[simChanges],[21])

    model.train(100,16)

    prices = items.getPrices(item)

    model.graphLoss()
    model.graphMAE()
    print(model.getScore())

    print("prices lengths same", len(prices), len(model.y_train) + len(model.y_test) + len(model.y_val))

    # try to optimize out small predictions
    test_prices = prices[len(model.y_train):len(model.y_train) + len(model.y_val)]
    budget = test_prices[0] * 101 - 1
    y_pred = model.predict(model.x_val)

    print(len(test_prices), len(y_pred))

    best = [-10000, 0, 0]
    for buySig in np.linspace(0, 1, 20):
        for sellSig in np.linspace(-1, 0, 20):
            buySigs = [(y_pred[i]-y_pred[i-3])/y_pred[i-3] >= buySig for i in range(3, len(y_pred)-3)]
            buySigs = [False] * 3 + buySigs + [False]*3
            for i in range(len(buySigs)):
                if buySigs[i]:
                    for j in range(1, 3):
                        buySigs[i + j] = False
            sellSigs = [(y_pred[i]-y_pred[i-3])/y_pred[i-3] <= sellSig for i in range(3, len(y_pred)-3)]
            sellSigs = [False] * 3 + sellSigs + [False]*3
            for i in range(len(sellSigs)):
                if sellSigs[i]:
                    for j in range(1, 3):
                        sellSigs[i + j] = False
            profit = ts.modelProfit(buySigs, sellSigs, test_prices, budget)[-1]
            if profit > best[0]:
                best = [profit, buySig, sellSig]

    test_prices = prices[-1 * len(model.y_test):]
    budget = test_prices[0] * 101 - 1
    y_pred = model.predict(model.x_test)

    scaler = StandardScaler()
    scaler.fit(np.array(test_prices).reshape(-1,1))
    y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1,1))

    plt.plot(scale(test_prices[2:]))
    plt.plot(scale(y_pred))
    plt.show()

    buySigs = [(y_pred[i]-y_pred[i-3])/y_pred[i-3] >= 0 for i in range(3, len(y_pred)-3)]
    buySigs = [False] * 3 +buySigs + [False] * 3
    for i in range(len(buySigs)):
        if buySigs[i]:
            for j in range(1, 3):
                buySigs[i + j] = False
    sellSigs = [(y_pred[i]-y_pred[i-3])/y_pred[i-3] <= 0 for i in range(3, len(y_pred)-3)]
    sellSigs = [False] * 3 + sellSigs + [False] * 3
    for i in range(len(sellSigs)):
        if sellSigs[i]:
            for j in range(1, 3):
                sellSigs[i + j] = False
    print("length check", len(test_prices),len(buySigs),len(sellSigs))
    profit_opt = ts.modelProfit(buySigs, sellSigs, test_prices, budget)

    buySigs = [(y_pred[i]-y_pred[i-3])/y_pred[i-3] >= best[1] for i in range(3, len(y_pred)-3)]
    buySigs = [False] * 3 + buySigs + [False] * 3
    for i in range(len(buySigs)):
        if buySigs[i]:
            for j in range(1, 3):
                buySigs[i + j] = False
    sellSigs = [(y_pred[i]-y_pred[i-3])/y_pred[i-3] <= best[2] for i in range(2, len(y_pred)-3)]
    sellSigs = [False] * 3 + sellSigs + [False] * 3
    for i in range(len(sellSigs)):
        if sellSigs[i]:
            for j in range(1, 3):
                sellSigs[i + j] = False
    profit = ts.modelProfit(buySigs, sellSigs, test_prices, budget)

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



    print(perf[-1], pers[-1], BaH[-1])
    print(profit[-1], profit_opt[-1], best[1], best[2])