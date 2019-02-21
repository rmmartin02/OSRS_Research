def perfectProfit(data,buyLimit,budget):
    init = budget
    buyLimit = int(buyLimit)
    invent = 0
    profits = []
    for i in range(len(data)-1):
        if data[i+1]>data[i]:
            buy = budget//data[i]
            if buy>buyLimit:
                buy = buyLimit
            invent += buy
            budget -= buy * data[i]
        else:
            budget += invent * data[i]
            invent = 0
        profits.append(((budget + invent * data[i]) - init) / init)
    return profits

def buyAndHold(data,buyLimit,budget):
    init = budget
    buyLimit = int(buyLimit)
    invent = 0
    profits = []
    buy = budget//data[0]
    if buy>buyLimit:
        buy = buyLimit
    invent += buy
    budget -= buy * data[0]
    for i in range(len(data)):
        profits.append(((budget+invent * data[i])-init)/init)
    return profits


def modelProfit(thresh,y_pred,data,buyLimit,budget):
    init = budget
    buyLimit = int(buyLimit)
    invent = 0
    profits = []
    for i in range(len(y_pred)):
        y = y_pred[i]
        if y-0.5>thresh:
            buy = budget//data[i]
            if buy>buyLimit:
                buy = buyLimit
            invent += buy
            budget -= buy * data[i]
        if y<0.5:
            budget += invent * data[i]
            invent = 0
        profits.append(((budget + invent * data[i]) - init) / init)
    return profits

def persistanceProfit(data,buyLimit,budget):
    init = budget
    buyLimit = int(buyLimit)
    invent = 0
    profits = []
    for i in range(1,len(data)):
        if data[i]>data[i-1]:
            buy = budget//data[i]
            if buy>buyLimit:
                buy = buyLimit
            invent += buy
            budget -= buy * data[i]
        else:
            budget += invent * data[i]
            invent = 0
        profits.append(((budget + invent * data[i]) - init) / init)
    return profits

def crossOverProfit(ind, sig, data,buyLimit,budget):
    init = budget
    buyLimit = int(buyLimit)
    invent = 0
    profits = []
    for i in range(-1 * len(sig) + 1, -1, 1):
        if ind[i - 1] > sig[i - 1] and ind[i] < sig[i]:
            budget += invent * data[i]
            invent = 0
        if ind[i - 1] < sig[i - 1] and ind[i] > sig[i]:
            buy = budget // data[i]
            if buy > buyLimit:
                buy = buyLimit
            invent += buy
            budget -= buy * data[i]
        profits.append(((budget + invent * data[i]) - init) / init)
    return profits