import matplotlib.pyplot as plt
import numpy as np
import pickle
import util.trading_systems as ts
import sys
import os
address = (os.sep).join(os.getcwd().split(os.sep)[:-2])
print(address)
sys.path.append(address)
print(sys.path)
import util.items as items
import keras
from keras.layers import InputLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

if __name__ == "__main__":

    item = "Amulet_of_glory(4)"

    similar = items.getSimilarItems(item)
    print(similar)
    similar = [a[0] for a in similar]
    similar = similar[:]
    print(similar)

    items.getPrices(item)[-10:]


    items.getInfo(item)

    ma = 21
    prices = items.getPrices(item)
    changes = items.getPriceChanges(item)
    quant = items.getQuants(item)
    print(len(prices),len(quant))

    ma12 = items.sma(prices,12)
    ma3 = items.sma(prices,3)[-1*len(ma12):]
    print(len(ma3),len(ma12))

    smaSell = []
    smaBuy = []
    for i in range(1,len(ma12)):
        if ma3[i-1]>ma12[i-1] and ma3[i]<ma12[i]:
            smaSell.append(i)
        if ma3[i-1]<ma12[i-1] and ma3[i]>ma12[i]:
            smaBuy.append(i)

    smaSignals = [0]*len(ma12)
    for a in smaSell:
        smaSignals[a] = -1
    for a in smaBuy:
        smaSignals[a] = 1

    kSlow, D = items.stochOscil(prices)
    kSlow = kSlow[-1*len(D):]

    stochSell = []
    stochBuy = []
    for i in range(1,len(D)):
        if kSlow[i]<20 and kSlow[i-1]>20:
            stochBuy.append(i)
        elif kSlow[i]<80 and D[i]<80 and kSlow[i-1]<D[i-1] and kSlow[i]>D[i]:
            stochBuy.append(i)
        if kSlow[i-1]>kSlow[i] and kSlow[i-1]>90:
            stochSell.append(i)
    print(len(stochBuy),len(stochSell))
    stochSignals = [0]*len(D)
    for a in stochSell:
        stochSignals[a] = -1
    for a in stochBuy:
        stochSignals[a] = 1

    mom, momMA = items.momentum(prices)
    mom = mom[-1*len(momMA):]

    momSell = []
    momBuy = []
    for i in range(1,len(momMA)):
        if mom[i-1]>momMA[i-1] and mom[i]<momMA[i]:
            momSell.append(i)
        if mom[i-1]<momMA[i-1] and mom[i]>momMA[i]:
            momBuy.append(i)

    momSignals = [0]*len(momMA)
    for a in momSell:
        momSignals[a] = -1
    for a in momBuy:
        momSignals[a] = 1

    ema9, macd = items.macd(prices)
    ema9 = ema9[-1*len(macd):]

    macdSell = []
    macdBuy = []
    for i in range(1,len(macd)):
        if ema9[i-1]>macd[i-1] and ema9[i]<macd[i]:
            macdSell.append(i)
        if ema9[i-1]<macd[i-1] and ema9[i]>macd[i]:
            macdBuy.append(i)

    macdSignals = [0]*len(macd)
    for a in macdSell:
        macdSignals[a] = -1
    for a in macdBuy:
        macdSignals[a] = 1

    def classify(d):
        if d>=0:
            return 1
        if d<0:
            return 0

    momEMA = items.ema(momSignals,3)
    smaEMA = items.ema(smaSignals,3)
    stochEMA = items.ema(stochSignals,3)
    macdEMA = items.ema(macdSignals,3)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1,1))

    features = [np.random.rand(len(changes)),changes,ma3,ma12,ema9,macd,mom,momMA,kSlow,D,momSignals,smaSignals,stochSignals,macdSignals]

    testModels = {}

    for f1 in range(1):
        fsizes = [0,10,0,0,0,0,0,0,10,0,10,0,0,0]
        #fsizes[f1] = 10

        x = []
        y = []
        p = []
        size = sum(fsizes)

        fsMax = max(fsizes)

        fMin = len(features[0])
        for f in features:
            if len(f)<fMin:
                fMin=len(f)

        for i in range(len(features)):
            features[i] = features[i][-1*fMin:]

        s1 = int(len(features[0])*.8)
        s2  =int(len(features[0])*.9)
        for i in range(len(features)):
            features[i][:s2] = list(scaler.fit_transform(np.array(features[i][:s2]).reshape(-1, 1)).reshape(len(features[i][:s2]),))
            features[i][s2:] = list(scaler.fit_transform(np.array(features[i][s2:]).reshape(-1, 1)).reshape(len(features[i][s2:]),))

        for i in range(-1*fMin+fsMax,0,1):
            xelem = []
            for j in range(len(features)):
                xelem = xelem + list(features[j][i-fsizes[j]:i])
            x.append(xelem)
            y.append(classify(changes[i]))
            p.append(prices[i])
        print(x[0],x[-1])
        print(len(x[0]),len(x[-1]))
        print(y[0],y[-1])

        x_train = np.array(x[:s1])
        y_train = np.array(y[:s1])

        x_val = np.array(x[s1:s2])
        y_val = np.array(y[s1:s2])
        p_val = np.array(p[s1:s2])

        x_test = np.array(x[s2:])
        y_test = np.array(y[s2:])
        p_test = np.array(p[s2:])

        print(len(x_train),len(y_train),len(x_val),len(y_val),len(x_test),len(y_test),len(p_test))
        print(len(x_train[0]),x_train[0])
        print(y_train[0])
        print(len(x_test[0]),x_test[0])
        print(y_test[0])

        model = Sequential()
        model.add(Dense(int(size), input_dim=size, activation='sigmoid'))
        model.add(Dense(int(size), activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        print(model.summary())

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                    validation_data=(x_val,y_val),
                    epochs=30,
                    batch_size=16)

        score = model.evaluate(x_test, y_test, batch_size=1)
        print(model.metrics_names)
        print(score)

        '''
        y_pred = model.predict(x_test)
        true_pos = len([y_pred[i] for i in range(len(y_pred)) if y_pred[i]>.5 and y_test[i]>.5])
        false_pos = len([y_pred[i] for i in range(len(y_pred)) if y_pred[i]>.5 and y_test[i]<.5])
        true_neg = len([y_pred[i] for i in range(len(y_pred)) if y_pred[i]<.5 and y_test[i]<.5])
        false_neg = len([y_pred[i] for i in range(len(y_pred)) if y_pred[i]<.5 and y_test[i]>.5])
        print(true_pos,false_pos,true_neg,false_neg,true_pos+true_neg+false_pos+false_neg)
        print('Accuracy: ', (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg))
        prec = (true_pos)/(true_pos+false_pos)
        print('Precision: ', prec)
        recall = (true_pos)/(true_pos+false_neg)
        print('Recall: ', recall)
        print('F1: ', (2*recall*prec)/(recall+prec))

        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'],label='Test')
        plt.plot(history.history['val_acc'],label='Valid')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'],label='Test Loss')
        plt.plot(history.history['val_loss'],label='Val Loss')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
        '''

        print(y_test[:20])
        print(p_test[:20])
        '''
        y_pred = model.predict(x_test)
        plt.plot([0, len(y_pred)], [.5, .5])
        plt.plot(y_pred,label='Pred')
        plt.plot(y_test,label='Actual')
        plt.legend()
        plt.show()
        plt.plot([0, len(y_test)], [.5, .5])
        plt.scatter(range(len(y_test)),y_pred,label='Pred')
        plt.scatter(range(len(y_test)),y_test,label='Actual')
        plt.legend()
        plt.show()
        '''

        bl = int(items.getInfo(item)['buyLimit'])
        budget = bl*p_test[0]+1
        print(bl,budget)

        y_pred = [classify(a) for a in model.predict(x_test)]
        perf = ts.perfectProfit(p_test,bl,budget)
        bAH = ts.buyAndHold(p_test,bl,budget)
        mod = ts.modelProfit(0,y_pred,p_test,bl,budget)
        pers = ts.persistanceProfit(p_test,bl,budget)
        momPr = ts.modelProfit(0,momSignals,p_test,bl,budget)

        print('Profits')
        print('Perfect:\t', perf[-1] * 100)
        print('Buy & Hold:\t', bAH[-1] * 100)
        print('Model:\t\t', mod[-1] * 100)
        print('Persistence:\t', pers[-1] * 100)
        print('Crossover:\t', momPr[-1] * 100)
        testModels[str(fsizes)] = score[1]

    sorted_d = sorted(testModels.items(), key=lambda kv: kv[1])
    print(sorted_d)
