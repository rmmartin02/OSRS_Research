import pickle

with open("0price2.pickle",'rb') as f:
	a = pickle.load(f)

with open("1price2.pickle",'rb') as f:
        b = pickle.load(f)

with open("2price2.pickle",'rb') as f:
        c = pickle.load(f)

with open("3price2.pickle",'rb') as f:
        d = pickle.load(f)

x = {**a, **b}
y = {**c, **d}

z = {**x, **y}

with open("all_price2.pickle", 'wb') as f:
	pickle.dump(z,f)
