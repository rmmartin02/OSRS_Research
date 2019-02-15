def perfectProfit(changes,data,buyLimit,budget):
    init = budget
    buyLimit = int(buyLimit)
    invent = 0
    profits = []
    for i in range(1,len(changes)):
        if changes[i]==1:
            buy = budget//data[i-1]
            if buy>buyLimit:
                buy = buyLimit
            invent += buy
            budget -= buy * data[i-1]
        else:
            budget += invent * data[i-1]
            invent = 0
        profits.append(((budget + (invent*data[i-1]))-init)/init)
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

def persistanceProfit(changes,data,buyLimit,budget):
    init = budget
    buyLimit = int(buyLimit)
    invent = 0
    profits = []
    for i in range(1,len(changes)):
        if changes[i-1]==1:
            buy = budget//data[i]
            if buy>buyLimit:
                buy = buyLimit
            invent += buy
            budget -= buy * data[i]
        else:
            budget += invent * data[i]
            invent = 0
        profits.append(((budget + invent * data[i]) - init) / init)
    profits.append(((budget + invent * data[i]) - init) / init)
    return profits