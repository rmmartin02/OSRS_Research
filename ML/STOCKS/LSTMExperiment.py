import sys
import os
address = (os.sep).join(os.getcwd().split(os.sep)[:-2])
print(address)
sys.path.append(address)
print(sys.path)
import util.items as items
import pickle
import util.trading_systems as ts
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from yahoofinancials import YahooFinancials


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

    return arr

def main():

    num = int(sys.argv[1])

    gpu = sys.argv[2]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # get list of items (probably divide this up)
    with open("Data/list{}".format(num), "r") as f:
        itemList = f.readlines()


    info = {}
    try:
        with open("Results/{}LSTM.pickle".format(num),'rb') as f:
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

                y = YahooFinancials(item).get_historical_price_data('2014-01-01', '2019-04-02', 'daily')
                prices = np.array([a['adjclose'] for a in y[item]['prices']])

                if len(prices) >= 1200:
                    prices = prices[-1200:]
                    print(len(prices))
                    #changes = items.getPriceChanges(prices)
                    toWrite['numPrices'] = len(prices)
                    # load the dataset
                    dataset = np.array(prices).reshape(-1, 1)
                    # normalize the dataset
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    dataset = scaler.fit_transform(dataset)
                    # split into train and test sets
                    train_size = int(len(dataset) * 0.8)
                    val_size = int(len(dataset) * 0.1)
                    test_size = len(dataset) - train_size - val_size
                    train, val, test = dataset[0:train_size, :], dataset[train_size:train_size + val_size, :], dataset[
                                                                                                               train_size + val_size:len(
                                                                                                                   dataset),
                                                                                                               :]
                    print(len(train), len(val), len(test))
                    # reshape into X=t and Y=t+1
                    look_back = 1
                    trainX, trainY = create_dataset(train, look_back)
                    valX, valY = create_dataset(val, look_back)
                    testX, testY = create_dataset(test, look_back)
                    print(len(testX), len(testY))
                    # reshape input to be [samples, time steps, features]
                    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                    valX = np.reshape(valX, (valX.shape[0], 1, valX.shape[1]))
                    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
                    # create and fit the LSTM network
                    print("training")
                    model = Sequential()
                    model.add(LSTM(4, input_shape=(1, look_back)))
                    model.add(Dense(1))
                    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
                    es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

                    beforeScore = model.evaluate(testX, testY, batch_size=1)
                    print(beforeScore)
                    toWrite['startLoss'] = beforeScore[0]
                    toWrite['startMAE'] = beforeScore[1]

                    history = model.fit(trainX, trainY, epochs=100, batch_size=16, validation_data=(valX, valY), callbacks=[es])

                    afterScore = model.evaluate(testX, testY, batch_size=1)
                    toWrite['endLoss'] = afterScore[0]
                    toWrite['endMAE'] = afterScore[1]
                    toWrite['numEpochs'] = len(history.history['loss'])

                    testPredict = model.predict(testX)
                    testPredict = scaler.inverse_transform(testPredict)
                    testY = scaler.inverse_transform(np.array(testY).reshape(-1, 1))

                    print(len(testY))
                    print(len(testPredict))

                    budget = 100

                    buySigs = [testY[i + 1] >= testY[i] for i in range(0, len(testY) - 1)]
                    buySigs = buySigs + [False]
                    sellSigs = [testY[i + 1] <= testY[i] for i in range(0, len(testY) - 1)]
                    sellSigs = sellSigs + [False]
                    print("lengths", len(buySigs), len(sellSigs), len(testY))
                    perf = ts.modelProfit(buySigs, sellSigs, testY, budget)

                    buySigs = [testY[i] >= testY[i - 1] for i in range(1, len(testY))]
                    buySigs = [False] + buySigs
                    sellSigs = [testY[i] <= testY[i - 1] for i in range(1, len(testY))]
                    sellSigs = [False] + sellSigs
                    print("lengths", len(buySigs), len(sellSigs), len(testY))
                    pers = ts.modelProfit(buySigs, sellSigs, testY, budget)

                    BaH = [(testY[i] / testY[0]) - 1 for i in range(len(testY))]

                    smaProf = ts.crossOverProfit(items.sma(testY, 3), items.sma(testY, 12), testY,
                                                 budget)
                    stchOsc = items.stochOscil(testY, 3, 5)
                    stchOscProf = ts.crossOverProfit(stchOsc[0], stchOsc[1], testY, budget)
                    mom = items.momentum(testY, 10)
                    momProf = ts.crossOverProfit(mom[0], mom[1], testY, budget)

                    buySigs = [testPredict[i] >= testPredict[i - 1] for i in range(1, len(testPredict))]
                    buySigs = [False] + buySigs
                    sellSigs = [testPredict[i] < testPredict[i - 1] for i in range(1, len(testPredict))]
                    sellSigs = [False] + sellSigs
                    print("lengths", len(buySigs), len(sellSigs), len(testY))
                    profit = ts.modelProfit(buySigs, sellSigs, testY, budget)

                    smaProf_Pred = ts.crossOverProfit(items.sma(testPredict, 3), items.sma(testPredict, 12), testY,
                                                      budget)
                    stchOsc = items.stochOscil(testPredict, 3, 5)
                    stchOscProf_Pred = ts.crossOverProfit(stchOsc[0], stchOsc[1], testY, budget)
                    mom = items.momentum(testPredict, 10)
                    momProf_Pred = ts.crossOverProfit(mom[0], mom[1], testY, budget)

                    toWrite['numItems'] = budget
                    toWrite['testPrices'] = testY
                    toWrite['predictions'] = testPredict

                    toWrite['model'] = profit[-1]
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
                #print(info)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    sys.exit()
                print('error',e)
                pass

            if count%25 == 0:
                with open('Results{}{}LSTM.pickle'.format(os.sep,num), 'wb') as f:
                    pickle.dump(info,f)
            count+=1

    with open('Results{}{}LSTM.pickle'.format(os.sep, num), 'wb') as f:
        pickle.dump(info, f)

def createList():
    keys = list(items.itemPrices.keys())
    for i in range(4):
        s = len(items.itemPrices)//4
        with open("Data/list" + str(i), 'w') as f:
            for j in range(s):
                f.write(keys[i*s+j]+"\n")

if __name__ == "__main__":
    main()