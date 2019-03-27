from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class RegressionModel:
    def __init__(self,labels,features,featuresizes, activation, layer1, layer2, split1, split2):
        self.changeData(labels,features,featuresizes,split1,split2)
        self.changeModel(layer1,layer2,activation)
        self.history = None


    def train(self,e,b):
        es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
        self.history = self.model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_val, self.y_val),
            epochs=e,
            batch_size=b,
            callbacks=[es]
            )

    def changeModel(self,layer1,layer2,activation):
        self.model = Sequential()
        self.model.add(Dense(layer1, input_dim=self.size, activation=activation))
        if layer2>0:
            self.model.add(Dense(layer2, activation=activation))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse',
                           optimizer='adam',
                           metrics=['mae'])


    def changeData(self,labels,features,featuresizes,split1,split2):
        self.x = []
        self.y = []
        self.features = features
        self.featuresizes = featuresizes

        self.size = sum(self.featuresizes)
        fsMax = max(self.featuresizes)

        # get feature list with smallest length so we can scale everything else
        fMin = len(self.features[0])
        for f in self.features:
            if len(f) < fMin:
                fMin = len(f)

        self.features = self.features
        for i in range(len(self.features)):
            self.features[i] = self.features[i][-1 * fMin:]

        scaler = StandardScaler()

        s1 = int(len(self.features[0]) * split1)
        s2 = int(len(self.features[0]) * split2)
        for i in range(len(self.features)):
            self.features[i][:s2] = list(
                scaler.fit_transform(np.array(self.features[i][:s2]).reshape(-1, 1)).reshape(
                    len(self.features[i][:s2]), ))
            self.features[i][s2:] = list(
                scaler.fit_transform(np.array(self.features[i][s2:]).reshape(-1, 1)).reshape(
                    len(self.features[i][s2:]), ))

        for i in range(-1 * fMin + fsMax, 0, 1):
            xelem = []
            for j in range(len(self.features)):
                xelem = xelem + list(self.features[j][i - featuresizes[j]:i])
            self.x.append(xelem)
            self.y.append(labels[i])

        self.y[:s2] = list(scaler.fit_transform(np.array(self.y[:s2]).reshape(-1, 1)).reshape(len(self.y[:s2]), ))
        self.y[s2:] = list(scaler.fit_transform(np.array(self.y[s2:]).reshape(-1, 1)).reshape(len(self.y[s2:]), ))

        self.x_train = np.array(self.x[:s1])
        self.y_train = np.array(self.y[:s1])

        self.x_val = np.array(self.x[s1:s2])
        self.y_val = np.array(self.y[s1:s2])

        self.x_test = np.array(self.x[s2:])
        self.y_test = np.array(self.y[s2:])


    def predict(self,x):
        return self.model.predict(x)


    def graphPredict(self):
        y_pred = self.model.predict(self.x_test)
        #scaler = MinMaxScaler(feature_range=(-1, 1))
        #y_pred = list(scaler.fit_transform(np.array(y_pred).reshape(-1, 1)).reshape(len(y_pred), ))
        plt.plot([0, len(y_pred)], [0, 0])
        plt.plot(y_pred, label='Pred')
        plt.plot(self.y_test, label='Actual')
        plt.legend()
        plt.show()


    def graphLoss(self):
        if self.history!=None:
            plt.plot(self.history.history['loss'],label='Test Loss')
            plt.plot(self.history.history['val_loss'],label='Val Loss')
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend()
            plt.show()


    def graphMAE(self):
        if self.history!=None:
            plt.plot(self.history.history['mean_absolute_error'],label='Test MAE')
            plt.plot(self.history.history['val_mean_absolute_error'],label='Val MAE')
            plt.title('model loss')
            plt.ylabel('mean_absolute_error')
            plt.xlabel('epoch')
            plt.legend()
            plt.show()


    def getModel(self):
        return self.model


    def getHistory(self):
        return self.history.history

    def getScore(self):
        return self.model.evaluate(self.x_test, self.y_test, batch_size=1)