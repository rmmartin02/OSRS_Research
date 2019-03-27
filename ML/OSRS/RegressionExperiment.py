import sys
import os
address = (os.sep).join(os.getcwd().split(os.sep)[:-2])
print(address)
sys.path.append(address)
print(sys.path)
import util.items as items
import pickle


def main():
    import util.trading_systems as ts
    import util.regression_model as rm
    import numpy as np

    num = int(sys.argv[1])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(num)

    # get list of items (probably divide this up)
    with open("ItemLists/list{}".format(num), "r") as f:
        itemList = f.readlines()


    info = {}
    try:
        with open("Results/{}.pickle".format(num),'rb') as f:
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
                changes = items.getPriceChanges(prices)
                toWrite['numPrices'] = len(prices)

                featSizes = [21]
                model = rm.RegressionModel(changes, [changes], featSizes, 'sigmoid', sum(featSizes), sum(featSizes), .8, .9)

                beforeScore = model.getScore()
                toWrite['startLoss'] = beforeScore[0]
                toWrite['startMAE'] = beforeScore[1]

                model.train(50, 16)

                afterScore = model.getScore()
                toWrite['endLoss'] = afterScore[0]
                toWrite['endMAE'] = afterScore[1]

                toWrite['numEpochs'] = len(model.getHistory()['loss'])

                bl = int(items.getInfo(item)['buyLimit'])
                test_prices = prices[-1 * len(model.y_test) + -1 * len(model.y_val):-1 * len(model.y_test)]
                budget = test_prices[0]*bl+1
                y_pred = model.predict(model.x_val)

                buySigs = [a >= 0 for a in y_pred]
                sellSigs = [a <= 0 for a in y_pred]
                profit = ts.modelProfit(buySigs, sellSigs, test_prices, bl, budget)[-1]
                best = [profit, 0, 0, len([a for a in buySigs if a == True]), len([a for a in sellSigs if a == True])]

                for buySig in np.linspace(-1, 1, 20):
                    for sellSig in np.linspace(1, -1, 20):
                        buySigs = [a >= buySig for a in y_pred]
                        sellSigs = [a <= sellSig for a in y_pred]
                        profit = ts.modelProfit(buySigs, sellSigs, test_prices, bl, budget)[-1]
                        if profit > best[0]:
                            best = [profit, buySig, sellSig, len([a for a in buySigs if a == True]),
                                    len([a for a in sellSigs if a == True])]

                test_prices = prices[-1 * len(model.y_test):]
                budget = test_prices[0]*bl+1
                y_pred = model.predict(model.x_test)

                toWrite['buyLimit'] = bl
                toWrite['budget'] = budget
                toWrite['testPrices'] = test_prices
                temp =  y_pred.tolist()
                toWrite['predictions'] = temp

                buySigs = [a >= best[1] for a in y_pred]
                sellSigs = [a <= best[2] for a in y_pred]
                profit = ts.modelProfit(buySigs, sellSigs, test_prices, bl, budget)
                best = [profit[-1], best[1], best[2], len([a for a in buySigs if a == True]), len([a for a in sellSigs if a == True])]

                toWrite['buySignal'] = best[1]
                toWrite['sellSignal'] = best[2]
                toWrite['numBuys'] = best[3]
                toWrite['numSells'] = best[4]

                toWrite['model'] = profit[-1]

                perf, pers, BaH = ts.baselines(test_prices, bl, budget)
                toWrite['perfect'] = perf[-1]
                toWrite['persist'] = pers[-1]
                toWrite['buyAndHold'] = BaH[-1]

                info[item] = toWrite
            except:
                pass

            if count%25 == 0:
                with open('Results{}{}.pickle'.format(os.sep,num), 'wb') as f:
                    pickle.dump(info,f)
            count+=1

    with open('Results{}{}.pickle'.format(os.sep, num), 'wb') as f:
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