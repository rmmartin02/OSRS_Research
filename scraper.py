from bs4 import BeautifulSoup
import requests
import re
import copy

URL = 'https://oldschool.runescape.wiki'

def getItems():
    nextURL = '/w/Category:Grand_Exchange_items'
    items = []
    done = False
    while not done:
        print(nextURL)
        r = requests.get(URL+nextURL)
        soup = BeautifulSoup(r.text,features="html.parser")
        arr = soup.find_all('li')
        i = 0
        while 'Category' not in arr[i].a['href']:
            items.append((arr[i].a['title'].replace(' ','_'),arr[i].a['href']))
            i+=1
        aList = soup.find_all('a')
        done = True
        for a in aList:
            if 'next page' in a.text:
                done = False
                nextURL = a['href']
    with open('itemURLs.csv','w') as f:
        for item in items:
            f.write('{},{}\n'.format(item[0],item[1]))

def convertYN(yn):
    if 'Yes':
        return True
    return False

def getItemInfo(item):
    r = requests.get(URL + '/w/' + item)
    soup = BeautifulSoup(r.text, features="html.parser")
    infoBox = soup.find_all('div',{'class':'infobox-wrapper'})[0]
    arr = []
    #if len(infoBox.find_all('div',{'class':'infobox-buttons'}))>0:
        #variants = infoBox.find_all('div', {'class': 'infobox-buttons'})[0].contents
        #for v in variants:
           # arr.append(parseItemInfo(item+v['data-switch-anchor']))
   # else:
       # arr.append(parseItemInfo(item))
    arr.append(parseItemInfo(item))
    return arr

def parsePrice(priceString):
    if 'Not sold' in priceString:
        return -1
    if priceString == '-':
        return -1
    return int(re.sub("\D","",priceString))

def parseVariants(item):
    print(item)
    r = requests.get(URL + '/w/' + item)
    soup = BeautifulSoup(r.text, features="html.parser")
    infoBox = soup.find_all('div', {'class': 'infobox-wrapper'})[0]
    buttons = infoBox.find_all('div', {'class': 'infobox-buttons'})
    if len(buttons)>0:
        hiddenInfo = infoBox.find_all('div', {'class': 'infobox-switch-resources hidden'})[0]
        print(hiddenInfo)
        with open('info.html','w') as f:
            f.write(str(hiddenInfo).replace('>','>\n'))
        names = []
        try:
            names = [n.text.replace(' ','_') for n in hiddenInfo.find_all('span',{'data-attr-param':'name'})[0].contents]
            print(names)
        except:
            pass
        releases = []
        updates = []
        try:
            dates = hiddenInfo.find_all('span',{'data-attr-param':'release'})[0]
            releases = [n.text.replace(' (Update)', '') for n in dates.contents[1:]]
            updates = [n['href'] for n in dates.find_all('a',text='Update')]
            print(releases)
            print(updates)
        except IndexError:
            pass
        storePrices = []
        try:
            storePrices = [parsePrice(n.text) for n in hiddenInfo.find_all('span',{'data-attr-param':'store'})[0].contents[1:]]
            print(storePrices)
        except IndexError:
            pass
        exchange = hiddenInfo.find_all('span',{'data-attr-param':'exchange'})[0]
        exURLs = {}
        for n in exchange.find_all('a'):
            exURLs[n['title'].split(':')[1].replace(' ','_')] = n['href'].split(':')[1]
        print(exURLs)

        defInfo = parseItemInfo([key for key in exURLs][0])
        infoDicts = []
        #only incldue items in the GE
        for it in exURLs:
            #stil working on this
            info = copy.deepcopy(defInfo)
            try:
                i = names.index(it)
                info['name'] = names[i]
            except:
                pass
            if len(releases)>i:
                info['released'] = releases[i]
            if len(updates) > i:
                info['update'] = updates[i]
            if len(storePrices) > i:
                info['storePrice'] = storePrices[i]
            exInfo = getExchangeInfo(exURLs[it])
            info['highAlch'] = exInfo['hialch']
            info['lowAlch'] = exInfo['lowalch']
            info['exchangePrice'] = exInfo['price']
            info['buyLimit'] = exInfo['limit']
            infoDicts.append(info)
        return infoDicts
    else:
        return [parseItemInfo(item)]


def parseItemInfo(item):
    r = requests.get(URL + '/w/' + item)
    soup = BeautifulSoup(r.text, features="html.parser")
    infoBox = soup.find_all('div',{'class':'infobox-wrapper'})[0]
    info = {}
    info['name'] =  item
    info['released'] = infoBox.find_all("th", text="Released")[0].parent.td.text[:]
    info['update'] = 'N/A'
    if 'Update' in infoBox.find_all("th", text="Released")[0].parent.td.text[8:]:
        info['released'] = info['released'].replace(' (Update)','')
        info['update'] = infoBox.find_all("th", text="Released")[0].parent.td.find_all('a')[-1]['href']

    info['members'] = convertYN(infoBox.find_all("th", text="Members")[0].parent.td.text)
    info['questItem'] = convertYN(infoBox.find_all("th", text="Quest item")[0].parent.td.text[:])
    info['tradeable'] = convertYN(infoBox.find_all("th", text="Tradeable")[0].parent.td.text[:])
    info['equipable'] = convertYN(infoBox.find_all("th", text="Equipable")[0].parent.td.text[:])
    info['stackable'] = convertYN(infoBox.find_all("th", text="Stackable")[0].parent.td.text[:])
    if len(infoBox.find_all("th", text="Noteable"))==0:
        info['noteable'] = 'N/A'
    else:
        info['noteable'] = convertYN(infoBox.find_all("th", text="Noteable")[0].parent.td.text[:])
    info['destroy'] = infoBox.find_all("th", text="Destroy")[0].parent.td.text[:]
    info['highAlch'] = int(re.sub("\D","",infoBox.find_all("th", text="High alch")[0].parent.td.text))
    info['lowAlch'] = int(re.sub("\D","",infoBox.find_all("th", text="Low alch")[0].parent.td.text))
    try:
        info['storePrice'] = int(re.sub("\D","",infoBox.find_all("th", text="Store price")[0].parent.td.text))
    except ValueError:
        info['storePrice'] = -1
    info['weight'] = float(infoBox.find_all("th", text="Weight")[0].parent.td.text[:-3])
    info['categories'] = [c.text for c in soup.find(id='catlinks').find_all('a') if 'href' in c.attrs and 'Category' in c['href']]
    exInfo = getExchangeInfo(item)
    info['highAlch'] = exInfo['hialch']
    info['lowAlch'] = exInfo['lowalch']
    info['exchangePrice'] = exInfo['price']
    info['buyLimit'] = exInfo['limit']
    return info

def storeItemInfo():
    items = []
    with open('itemURLs.csv', 'r') as f:
        lines = f.readlines()
        items = [i.split(',')[0] for i in lines]
    with open('itemsInfo.csv','w') as f:
        i = 0
        for i in range(len(items)):
            for info in parseVariants(items[i]):
                #print(info)
                info = [info['name'],
                        info['released'],
                        info['update'],
                        info['members'],
                        info['questItem'],
                        info['tradeable'],
                        info['equipable'],
                        info['stackable'],
                        info['noteable'],
                        info['destroy'],
                        info['highAlch'],
                        info['lowAlch'],
                        info['storePrice'],
                        info['exchangePrice'],
                        info['buyLimit'],
                        info['weight'],
                        info['categories']]
                info = [str(i) for i in info]
                f.write(','.join(info)+'\n')
            print('{}/{} ({})'.format(i,len(items),float(i)/float(len(items))))


def getExchangeInfo(item):
    item = item.replace('+','%2B')
    r = requests.get('{}/w/Module:Exchange/{}?action=raw'.format(URL, item))
    info = {}
    keys = ['price', 'value', 'limit', 'hialch', 'lowalch']
    for k in keys:
        info[k] = -1
    if len(r.text) == 0:
        return info
    arr = r.text.split('\n')
    print(arr)
    for a in arr:
        for k in keys:
            if k in a and 'examine' not in a:
                try:
                    info[k] = a.split('=')[1]
                except ValueError:
                    info[k] = -1
    print(info)
    return info

def getExchangePrices(item):
    r = requests.get('{}/w/Module:Exchange/{}/Data?action=raw'.format(URL,item))
    arr = r.text[8:-1].split()
    arr = [a.replace('"','').replace("'","").replace(',','').split(':') for a in arr]
    times = [int(t[0]) for t in arr]
    prices = [int(p[1]) for p in arr]
    return times,prices

if __name__ == "__main__":
    storeItemInfo()