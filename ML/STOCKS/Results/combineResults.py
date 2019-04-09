import pickle

with open("0change.pickle",'rb') as f:
	a = pickle.load(f)

with open("1change.pickle",'rb') as f:
        b = pickle.load(f)

with open("2change.pickle",'rb') as f:
        c = pickle.load(f)

with open("3change.pickle",'rb') as f:
        d = pickle.load(f)

x = {**a, **b}
y = {**c, **d}

z = {**x, **y}

with open("change.pickle", 'wb') as f:
	pickle.dump(z,f)
