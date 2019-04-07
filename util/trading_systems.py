def perfectProfit(data,buyLimit,numItems):
    invent = numItems
    init = invent*data[0]
    budget = 0
    profits = []
    for i in range(len(data)-1):
        if data[i+1]>data[i]:
            if invent>0:
                budget += invent * data[i]
                invent = 0
            buy = budget//data[i]
            if buy>buyLimit:
                buy = buyLimit
            invent += buy
            budget -= buy * data[i]
        else:
            if invent>0:
                budget += invent * data[i]
                invent = 0
        profits.append(((budget + invent * data[i]) - init) / init)
    return profits

def buyAndHold(data,buyLimit,numItems):
    invent = numItems
    init = invent*data[0]
    budget = 0
    profits = []
    buy = budget//data[0]
    if buy>buyLimit:
        buy = buyLimit
    invent += buy
    budget -= buy * data[0]
    for i in range(len(data)):
        profits.append(((budget+invent * data[i])-init)/init)
    return profits


def modelProfit(buy_signal,sell_signal,data,numItems):
    invent = numItems
    init = invent*data[0]
    budget = 0
    profits = []
    for i in range(len(data)):
        if buy_signal[i]:
            #sell first then buy
            if invent>0:
                budget += invent * data[i]
                invent = 0
            buy = budget//data[i]
            invent += buy
            budget -= buy * data[i]
        elif sell_signal[i] and invent>0:
            budget += invent * data[i]
            invent = 0
        profits.append(((budget + invent * data[i]) - init) / init)
    return profits

def persistanceProfit(data, numItems):
    invent = numItems
    init = invent*data[0]
    budget = 0
    profits = []
    for i in range(1,len(data)):
        if data[i]>data[i-1]:
            if invent>0:
                budget += invent * data[i]
                invent = 0
            buy = budget//data[i]
            invent += buy
            budget -= buy * data[i]
        elif invent>0:
            budget += invent * data[i]
            invent = 0
        profits.append(((budget + invent * data[i]) - init) / init)
    return profits

def crossOverProfit(ind, sig, data, numItems):
    invent = numItems
    init = invent*data[0]
    budget = 0
    profits = []
    for i in range(-1 * len(sig) + 1, -1, 1):
        if ind[i - 1] > sig[i - 1] and ind[i] < sig[i] and invent>0:
            budget += invent * data[i]
            invent = 0
        if ind[i - 1] < sig[i - 1] and ind[i] > sig[i]:
            if invent>0:
                budget += invent * data[i]
                invent = 0
            buy = budget // data[i]
            invent += buy
            budget -= buy * data[i]
        profits.append(((budget + invent * data[i]) - init) / init)
    return profits

def baselines(data,buyLimit,budget):
    return perfectProfit(data,buyLimit,budget),persistanceProfit(data,buyLimit,budget),buyAndHold(data,buyLimit,budget)