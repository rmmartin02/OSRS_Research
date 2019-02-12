def perfectProfit(x_arr,thresh,y_arr,data,buyLimit,budget):
    init = budget
    buyLimit = int(buyLimit)
    invent = 0
    profits = []
    for i in range(len(x_arr)):
        x = x_arr[i][-1]
        y_pred = y_arr[i]
        if y_pred-x>thresh:
            buy = budget//data[i]
            if buy>buyLimit:
                buy = buyLimit
            invent += buy
            budget -= buy * data[i]
        if y_pred<x:
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


def modelProfit(x_arr,thresh,model,data,buyLimit,budget):
    init = budget
    buyLimit = int(buyLimit)
    invent = 0
    profits = []
    for i in range(len(x_arr)):
        x = x_arr[i][-1]
        y_pred = model.predict(x_arr[i].reshape(1,-1))[0][0]
        if y_pred-x>thresh:
            buy = budget//data[i]
            if buy>buyLimit:
                buy = buyLimit
            invent += buy
            budget -= buy * data[i]
        if y_pred<x:
            budget += invent * data[i]
            invent = 0
        profits.append(((budget + invent * data[i]) - init) / init)
    return profits