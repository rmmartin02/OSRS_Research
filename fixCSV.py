with open('itemsInfo.csv','r') as f:
    lines = f.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].replace(',,',',')
    a = lines[i].index('[')
    cats = lines[i][a:]
    lines[i] = lines[i][:a].replace(',','\t')
    lines[i] = lines[i] + cats

with open('itemsInfo.tsv','w') as f:
    for l in lines:
        f.write(l+'\n')
