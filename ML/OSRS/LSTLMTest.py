# LSTM for international airline passengers problem with window regression framing
import sys
import os
address = (os.sep).join(os.getcwd().split(os.sep)[:-2])
print(address)
sys.path.append(address)
print(sys.path)
import util.items as items
import util.trading_systems as ts
import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
#dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
#dataset = dataframe.values
item = "Abyssal_whip"
dataset = numpy.array(items.getPrices(item)).reshape(-1,1)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size
train, val, test = dataset[0:train_size,:], dataset[train_size:train_size+val_size,:], dataset[train_size+val_size:len(dataset),:]
print(len(train),len(val),len(test))
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
valX, valY = create_dataset(val, look_back)
testX, testY = create_dataset(test, look_back)
print(len(testX),len(testY))
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
valX = numpy.reshape(valX, (valX.shape[0], 1, valX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
print("training")
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
model.fit(trainX, trainY, epochs=50, batch_size=16, validation_data=(valX, valY), callbacks=[es])

testPredict = model.predict(testX)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(numpy.array(testY).reshape(-1,1))

print(len(testY))
print(len(testPredict))

plt.plot(testY)
plt.plot(testPredict)
plt.show()

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

buySigs = [testPredict[i] >= testY[i - 1] for i in range(1, len(testPredict))]
buySigs = [False] + buySigs
sellSigs = [testPredict[i] < testY[i - 1] for i in range(1, len(testPredict))]
sellSigs = [False] + sellSigs
print("lengths", len(buySigs), len(sellSigs), len(testY))
profit = ts.modelProfit(buySigs, sellSigs, testY, budget)

buySigs = [testPredict[i] >= testPredict[i - 1] for i in range(1, len(testPredict))]
buySigs = [False] + buySigs
sellSigs = [testPredict[i] < testPredict[i - 1] for i in range(1, len(testPredict))]
sellSigs = [False] + sellSigs
print("lengths", len(buySigs), len(sellSigs), len(testY))
profit_pred = ts.modelProfit(buySigs, sellSigs, testY, budget)

smaProf_Pred = ts.crossOverProfit(items.sma(testPredict, 3), items.sma(testPredict, 12), testY, budget)
stchOsc = items.stochOscil(testPredict, 3, 5)
stchOscProf_Pred = ts.crossOverProfit(stchOsc[0], stchOsc[1], testY, budget)
mom = items.momentum(testPredict, 10)
momProf_Pred = ts.crossOverProfit(mom[0], mom[1], testY, budget)

print(perf[-1], pers[-1], BaH[-1])
print(profit[-1],profit_pred[-1])
print(smaProf[-1], stchOscProf[-1], momProf[-1])
print(smaProf_Pred[-1], stchOscProf_Pred[-1], momProf_Pred[-1])

'''
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
'''
