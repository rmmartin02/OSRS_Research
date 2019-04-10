import pickle

with open("0LSTM.pickle",'rb') as f:
	a = pickle.load(f)

with open("1LSTM.pickle",'rb') as f:
        b = pickle.load(f)

with open("2LSTM.pickle",'rb') as f:
        c = pickle.load(f)

with open("3LSTM.pickle",'rb') as f:
        d = pickle.load(f)

x = {**a, **b}
y = {**c, **d}

z = {**x, **y}

with open("LSTM.pickle", 'wb') as f:
	pickle.dump(z,f)
