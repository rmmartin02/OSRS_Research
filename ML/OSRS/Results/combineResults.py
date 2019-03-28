import pickle

with open("0.pickle",'rb') as f:
	a = pickle.load(f)

with open("1.pickle",'rb') as f:
        b = pickle.load(f)

with open("2.pickle",'rb') as f:
        c = pickle.load(f)

with open("3.pickle",'rb') as f:
        d = pickle.load(f)

x = {**a, **b}
y = {**c, **d}

z = {**x, **y}

with open("all.pickle", 'wb') as f:
	pickle.dump(z,f)
