import sys
import os
address = (os.sep).join(os.getcwd().split(os.sep)[:-2])
print(address)
sys.path.append(address)
print(sys.path)
import util.items as items
import pickle

def scale(p):
    m = max(p)
    n = min(p)
    if abs(n)>abs(m):
        m = abs(n)
    arr = [0.0] * len(p)
    for i in range(len(p)):
        arr[i] = p[i] / m
    return arr


def main():
    import util.trading_systems as ts
    import util.regression_model as rm
    import numpy as np

    num = int(sys.argv[1])

    gpu = sys.argv[2]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # get list of items (probably divide this up)
    with open("ItemLists/list{}".format(num), "r") as f:
        itemList = f.readlines()


    info = {}
    try:
        with open("Results/{}change.pickle".format(num),'rb') as f:
            info = pickle.load(f)
    except FileNotFoundError:
        pass

    count = 0
    for item in itemList:
        item = item.strip()
        print(item)
        if item not in info:
            print('Processing',item)
            # go through list training model for each item
            try:
                toWrite = {}

                toWrite['item'] = item

                prices = items.getPrices(item)
                if len(prices)>=1200:
                    prices = prices[-1200:]
                    print(len(prices))
                    changes = items.getPriceChanges(prices)
                    toWrite['numPrices'] = len(prices)

                    featSizes = [21]
                    model = rm.RegressionModel(scale(changes), [scale(changes)], featSizes, 'sigmoid', sum(featSizes), sum(featSizes), .8, .9)

                    beforeScore = model.getScore()
                    toWrite['startLoss'] = beforeScore[0]
                    toWrite['startMAE'] = beforeScore[1]

                    model.train(30, 16)

                    afterScore = model.getScore()
                    toWrite['endLoss'] = afterScore[0]
                    toWrite['endMAE'] = afterScore[1]

                    toWrite['numEpochs'] = len(model.getHistory()['loss'])

                    # try to optimize out small predictions
                    test_prices = prices[len(model.y_train):len(model.y_train) + len(model.y_val)]
                    numItems = 100
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
                            profit = ts.modelProfit(buySigs, sellSigs, test_prices, numItems)[-1]
                            if profit > best[0]:
                                best = [profit, buySig, sellSig]

                    test_prices = prices[-1 * len(model.y_test):]
                    numItems = 100
                    y_pred = model.predict(model.x_test)

                    print(len(test_prices), len(y_pred))

                    buySigs = [test_prices[i + 1] >= test_prices[i] for i in range(0, len(test_prices) - 1)]
                    buySigs = buySigs + [False]
                    sellSigs = [test_prices[i + 1] <= test_prices[i] for i in range(0, len(test_prices) - 1)]
                    sellSigs = sellSigs + [False]
                    print("lengths", len(buySigs), len(sellSigs), len(test_prices))
                    perf = ts.modelProfit(buySigs, sellSigs, test_prices, numItems)

                    buySigs = [test_prices[i] >= test_prices[i - 1] for i in range(1, len(test_prices))]
                    buySigs = [False] + buySigs
                    sellSigs = [test_prices[i] <= test_prices[i - 1] for i in range(1, len(test_prices))]
                    sellSigs = [False] + sellSigs
                    print("lengths", len(buySigs), len(sellSigs), len(test_prices))
                    pers = ts.modelProfit(buySigs, sellSigs, test_prices, numItems)

                    BaH = [(test_prices[i] / test_prices[0]) - 1 for i in range(len(test_prices))]

                    smaProf = ts.crossOverProfit(items.sma(test_prices, 3), items.sma(test_prices, 12), test_prices,
                                                 numItems)
                    stchOsc = items.stochOscil(test_prices, 3, 5)
                    stchOscProf = ts.crossOverProfit(stchOsc[0], stchOsc[1], test_prices, numItems)
                    mom = items.momentum(test_prices, 10)
                    momProf = ts.crossOverProfit(mom[0], mom[1], test_prices, numItems)

                    buySigs = [y_pred[i] >= 0 for i in range(1, len(y_pred))]
                    buySigs = [False] + buySigs
                    sellSigs = [y_pred[i] < 0 for i in range(1, len(y_pred))]
                    sellSigs = [False] + sellSigs
                    print("lengths", len(buySigs), len(sellSigs), len(test_prices))
                    profit = ts.modelProfit(buySigs, sellSigs, test_prices, numItems)

                    buySigs = [y_pred[i] >= best[1] for i in range(1, len(y_pred))]
                    buySigs = [False] + buySigs
                    sellSigs = [y_pred[i] - y_pred[i - 1] <= best[2] for i in range(1, len(y_pred))]
                    sellSigs = sellSigs + [False]
                    profit_opt = ts.modelProfit(buySigs, sellSigs, test_prices, numItems)

                    smaProf_Pred = ts.crossOverProfit(items.sma(y_pred, 3), items.sma(y_pred, 12), test_prices, numItems)
                    stchOsc = items.stochOscil(y_pred, 3, 5)
                    stchOscProf_Pred = ts.crossOverProfit(stchOsc[0], stchOsc[1], test_prices, numItems)
                    mom = items.momentum(y_pred, 10)
                    momProf_Pred = ts.crossOverProfit(mom[0], mom[1], test_prices, numItems)

                    print(perf[-1], pers[-1], BaH[-1])
                    print(profit[-1], profit_opt[-1], best[1], best[2])
                    print(smaProf[-1], stchOscProf[-1], momProf[-1])
                    print(smaProf_Pred[-1], stchOscProf_Pred[-1], momProf_Pred[-1])

                    toWrite['numItems'] = numItems
                    toWrite['testPrices'] = test_prices
                    temp =  y_pred.tolist()
                    temp = [a[0] for a in temp]
                    toWrite['predictions'] = temp

                    true_pos = len([y_pred[i] for i in range(len(y_pred)) if y_pred[i] > 0 and model.y_test[i] > 0])
                    false_pos = len([y_pred[i] for i in range(len(y_pred)) if y_pred[i] > 0 and model.y_test[i] < 0])
                    true_neg = len([y_pred[i] for i in range(len(y_pred)) if y_pred[i] < 0 and model.y_test[i] < 0])
                    false_neg = len([y_pred[i] for i in range(len(y_pred)) if y_pred[i] < 0 and model.y_test[i] > 0])
                    toWrite['Accuracy'] =  (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
                    prec = (true_pos) / (true_pos + false_pos + .0001)
                    toWrite['Precision'] = prec
                    recall = (true_pos) / (true_pos + false_neg + .0001)
                    toWrite['Recall'] = recall
                    toWrite['F1'] = (2 * recall * prec) / (recall + prec)

                    toWrite['model'] = profit[-1]
                    toWrite['model_opt'] = profit_opt[-1]
                    toWrite['opt_params'] = (best[1],best[2])
                    toWrite['perfect'] = perf[-1]
                    toWrite['persist'] = pers[-1]
                    toWrite['buyAndHold'] = BaH[-1]
                    toWrite['sma'] = smaProf[-1]
                    toWrite['sma_model'] = smaProf_Pred[-1]
                    toWrite['stochOscil'] = stchOscProf[-1]
                    toWrite['stochOscil_model'] = stchOscProf_Pred[-1]
                    toWrite['momentum'] = momProf[-1]
                    toWrite['momentum_model'] = momProf_Pred[-1]

                    info[item] = toWrite
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    sys.exit()
                print(e)
                pass

            if count%25 == 0:
                with open('Results{}{}change.pickle'.format(os.sep,num), 'wb') as f:
                    pickle.dump(info,f)
            count+=1

    with open('Results{}{}change.pickle'.format(os.sep, num), 'wb') as f:
        pickle.dump(info, f)

def createList():
    keys = list(items.itemPrices.keys())
    for i in range(4):
        s = len(items.itemPrices)//4
        with open("ItemLists/list" + str(i), 'w') as f:
            for j in range(s):
                f.write(keys[i*s+j]+"\n")

if __name__ == "__main__":
    main()