import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import linregress
import scraper
import random

def getY(a,b):
    if a-b>0:
        return 1
    return 0

def LogReg(X,y,sample,prices):
    #divide 80:20
    X_train = X[:int(len(X)*.8)]
    Y_train = y[:int(len(y)*.8)]
    X_test = X[int(len(X)*.8):]
    Y_test = y[int(len(y)*.8):]
    sample_test = sample[int(len(y)*.8):]
    #print(len(X_train),len(Y_train),len(X_test),len(Y_test))
    #print(X_train, Y_train, X_test, Y_test)
    logreg = LogisticRegression(solver='liblinear')
    logreg.fit(X_train,Y_train)
    Y_pred = logreg.predict(X_test)
    #print(Y_pred)

    train_error = logreg.score(X_train,Y_train)
    test_error = logreg.score(X_test,Y_test)

    #get average profit per purchase
    rand_profit = []
    pred_profit = []
    perf_profit = []
    for i in range(len(sample_test)):
        if Y_pred[i]==1:
            pred_profit.append(prices[sample_test[i]]-prices[sample_test[i]-1])
        if Y_test[i]==1:
            perf_profit.append(prices[sample_test[i]]-prices[sample_test[i]-1])
        if random.randint(0,1)==1:
            rand_profit.append(prices[sample_test[i]]-prices[sample_test[i]-1])
    pred_profit = np.mean(pred_profit)
    perf_profit = np.mean(perf_profit)
    rand_profit = np.mean(rand_profit)
    return train_error,test_error,pred_profit,perf_profit,rand_profit


if __name__ == "__main__":
    #train logistic regression using previous 20 price points to predict next one
    itemPrices = scraper.loadItemPrices()
    prices = [a[1] for a in itemPrices['Rune_boots']]
    train_errors = []
    test_errors = []
    pred_profits = []
    perf_profits = []
    rand_profits = []
    step = 10
    for i in range(20,len(prices)-20-step-1,step):
        X = []
        y = []
        sample = random.sample(range(20,len(prices)),i)
        for r in sample:
            diffs = []
            for j in range(1,2):
                diffs.append(getY(prices[r-j],prices[r-j-1]))
            X.append(diffs)
            y.append(getY(prices[r],prices[r-1]))
        train, test, pred_profit, perf_profit, rand_profit = LogReg(X,y,sample,prices)
        train_errors.append(train)
        test_errors.append(test)
        pred_profits.append(pred_profit)
        perf_profits.append(perf_profit)
        rand_profits.append(rand_profit)
    x = range(20,len(prices)-20-step-1,step)
    #plt.plot(x,train_errors)
    #plt.plot(x,test_errors)

    plt.plot(pred_profits,'b-')
    plt.plot(rand_profits,'r--')
    plt.plot(perf_profits,'g--')
    plt.show()

    '''
    (m, b) = np.polyfit(x, profits, 1)
    print(m, b)

    yp = np.polyval([m, b], x)
    plt.plot(x, yp)
    plt.grid(True)
    plt.scatter(x, profits)
    plt.show()
    '''