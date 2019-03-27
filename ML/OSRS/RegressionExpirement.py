import sys
import os
address = (os.sep).join(os.getcwd().split(os.sep)[:-2])
print(address)
sys.path.append(address)
print(sys.path)
import util.items as items


def main():
    import util.trading_systems as ts
    import util.regression_model as rm
    import numpy as np

    num = int(sys.argv[1])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(num)

    # get list of items (probably divide this up)
    with open("ItemLists/list{}".format(num), "r") as f:
        itemList = f.readlines()

    for item in itemList:
        # go through list training model for each item
        try:
            item = item.strip()
            toWrite = []

            toWrite.append(item)

            prices = items.getPrices(item)
            changes = items.getPriceChanges(prices)
            toWrite.append(len(prices))

            featSizes = [21]
            model = rm.RegressionModel(changes, [changes], featSizes, 'sigmoid', sum(featSizes), sum(featSizes), .8, .9)

            beforeScore = model.getScore()
            toWrite.append(beforeScore[0])
            toWrite.append(beforeScore[1])

            model.train(50, 16)

            afterScore = model.getScore()
            toWrite.append(afterScore[0])
            toWrite.append(afterScore[1])

            toWrite.append(len(model.getHistory()['loss']))

            bl = int(items.getInfo(item)['buyLimit'])
            test_prices = prices[-1 * len(model.y_test) + -1 * len(model.y_val):-1 * len(model.y_test)]
            budget = test_prices[0]
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
            budget = test_prices[0]
            y_pred = model.predict(model.x_test)

            toWrite.append(bl)
            toWrite.append(budget)
            toWrite.append(test_prices)
            toWrite.append(list(y_pred))

            buySigs = [a >= best[1] for a in y_pred]
            sellSigs = [a <= best[2] for a in y_pred]
            profit = ts.modelProfit(buySigs, sellSigs, test_prices, bl, budget)
            best = [profit[-1], best[1], best[2], len([a for a in buySigs if a == True]),
                    len([a for a in sellSigs if a == True])]

            toWrite.append(best[1])
            toWrite.append(best[2])
            toWrite.append(best[3])
            toWrite.append(best[4])

            toWrite.append(profit[-1])

            perf, pers, BaH = ts.baselines(test_prices, bl, budget)
            toWrite.append(perf[-1])
            toWrite.append(pers[-1])
            toWrite.append(BaH[-1])

            with open("Results/{}.tsv".format(num), 'a') as f:
                for a in toWrite:
                    f.write(str(a) + '\t')
                f.write('\n')
        except:
            pass

def createList():
    keys = list(items.itemPrices.keys())
    for i in range(4):
        s = len(items.itemPrices)//4
        with open("ItemLists/list" + str(i), 'w') as f:
            for j in range(s):
                f.write(keys[i*s+j]+"\n")

if __name__ == "__main__":
    main()