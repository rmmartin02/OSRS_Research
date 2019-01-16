import ast
import networkx as nx

with open('itemsInfo.tsv','r') as f:
    lines = f.readlines()
cats = []
for c in lines:
    a = c.split('\t')[-1].rstrip().replace('[','').replace(']','')
    try:
        cats.append(list(ast.literal_eval(a)))
    except:
        pass
print(cats)

catNames = []
count = 0
for l in cats:
    count+=1
    print(count,len(cats))
    for d in cats:
        for c in d:
            if c not in catNames:
                catNames.append(c)

print(catNames)

itemCounts = {}
for c in catNames:
    itemCounts[c] = 0

edgeCounts = []
for c in catNames:
    edgeCounts.append([0]*len(catNames))

for l in cats:
    for i in range(len(l)):
        itemCounts[l[i]] += 1
        for j in range(i,len(l)):
            row = catNames.index(l[i])
            col = catNames.index(l[j])
            edgeCounts[row][col] += 1

print(itemCounts)
for e in edgeCounts:
    print(e)

G = nx.Graph()
G.add_nodes_from(catNames)
G.add_edge(2, 3, weight=0.9)