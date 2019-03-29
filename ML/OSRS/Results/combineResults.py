import pickle

with open("0_1000.pickle",'rb') as f:
	a = pickle.load(f)

with open("1_1000.pickle",'rb') as f:
        b = pickle.load(f)

with open("2_1000.pickle",'rb') as f:
        c = pickle.load(f)

with open("3_1000.pickle",'rb') as f:
        d = pickle.load(f)

x = {**a, **b}
y = {**c, **d}

z = {**x, **y}

with open("all_1000.pickle", 'wb') as f:
	pickle.dump(z,f)
