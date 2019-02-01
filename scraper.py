from selenium import webdriver
from bs4 import BeautifulSoup
from pprint import pprint
import requests
import re
import pickle
import ast

URL = 'https://oldschool.runescape.wiki'
KEYS = ['name','image','released','update','members','quest','tradeable','equipable',
        'stackable','edible','noteable','destroy','storeprice','storeCurrency','seller',
        'alchable','highAlch','lowAlch','exchangePrice','buyLimit','weight','examine']

def convertYN(yn):
    if yn == 'Yes' or yn == 'yes':
        return True
    return False

def getVariants(item):
    r = requests.get(URL + '/w/' + item)
    soup = BeautifulSoup(r.text, features="html.parser")
    infoBox = soup.find('div',{'class':'infobox-wrapper'})
    arr = []
    if len(infoBox.find_all('div',{'class':'infobox-buttons'}))>0:
        variants = infoBox.find('div', {'class': 'infobox-buttons'}).contents
        for v in variants:
            arr.append(item+v['data-switch-anchor'])
    else:
        arr.append(item)
    return arr

def parsePrice(priceString):
    try:
        return int(re.sub("\D","",priceString))
    except ValueError:
        return -1

def getExchangeInfo(item):
    item = item.replace('+','%2B')
    r = requests.get('{}/w/Module:Exchange/{}?action=raw'.format(URL, item))
    info = {}
    keys = ['price', 'value', 'limit', 'hialch', 'lowalch','examine']
    for k in keys:
        info[k] = -1
    if len(r.text) == 0:
        return info
    arr = [a.split('=') for a in r.text.split('\n')[1:-1]]
    for a in arr:
        for k in keys:
            if k in a[0]:
                try:
                    if k != 'examine':
                        info[k] = int(re.sub("\D","",a[1]))
                    else:
                        info[k] = a[1][2:-2]
                except ValueError:
                    info[k] = -1
    return info


def parseItemInfo(item, browser):
    # https://oldschool.runescape.wiki/w/Template:Infobox_Item/doc
    request = URL + '/w/' + item
    browser.get(request)  # navigate to page behind login
    r = browser.execute_script("return document.body.innerHTML")  # returns the inner HTML as a string
    soup = BeautifulSoup(r, features="html.parser")
    infoBox = soup.find('div', {'class': 'infobox-wrapper'})
    with open('info.html', 'w') as f:
        f.write(str(infoBox).replace('>', '>\n'))
    info = {}
    for key in KEYS:
        info[key] = 'N/A'
    info['name'] = infoBox.find('th',{'class':'infobox-header'}).text.replace(' ','_')
    info['image'] = infoBox.find('td',{'class':'infobox-image inventory-image'}).a['href']
    release = infoBox.find("th", text = 'Released').parent.td
    info['released'] = release.text.replace(' (Update)','')
    info['update'] = release.find('a',text='Update')['href']
    '''Don't think aka is even used?
    try:
        info['aka'] = infoBox.find("td", {'data-attr-param':'aka'}).text
    except AttributeError:
        pass'''
    info['members'] = convertYN(infoBox.find("th", text = 'Members').parent.td.text)
    info['quest'] = infoBox.find("th", text = 'Quest item').parent.td.text
    info['tradeable'] = convertYN(infoBox.find("th", text = 'Tradeable').parent.td.text) #Yes No Yes - only when uncharged
    if info['tradeable']:
        exInfo = getExchangeInfo(info['name'])
        info['examine'] = exInfo['examine']
        info['exchangePrice'] = exInfo['price']
        info['buyLimit'] = exInfo['limit']
    try:
        info['bankable'] = convertYN(infoBox.find("th", text = 'Bankable').parent.td.text)
    except AttributeError:
        info['bankable'] = True
    '''Again don't think this ever really comes up
    try:
        info['stacksinbank'] = convertYN(infoBox.find("th", text = 'Members').parent.td.text)
    except AttributeError:
        info['stacksinbank'] = True'''
    info['equipable'] = convertYN(infoBox.find("th", text = 'Equipable').parent.td.text)
    info['stackable'] = convertYN(infoBox.find("th", text = 'Stackable').parent.td.text)
    try:
        info['edible'] = convertYN(infoBox.find("th", text = 'Edible').parent.td.text)
    except AttributeError:
        info['edible'] = False
    try:
        info['noteable'] = convertYN(infoBox.find("th", text = 'Noteable').parent.td.text)
    except AttributeError:
        info['noteable'] = True
    info['destroy'] = infoBox.find("th", text = 'Destroy').parent.td.text
    store = infoBox.find("th", text = 'Store price').parent.td #price currency store
    info['storePrice'] = parsePrice(store.text.split()[0])
    storeLinks = infoBox.find_all('a')
    if storeLinks==2:
        info['storeCurrency'] = storeLinks[0].text
        info['seller'] = storeLinks[1].text
    if storeLinks==1:
        info['storeCurrency'] = 'coins'
        info['seller'] = storeLinks[0].text
    try:
        info['alchable'] = convertYN(infoBox.find("th", text='Alchemy').parent.td.text)
    except AttributeError:
        info['alchable'] = True
    if info['alchable']:
        info['highAlch'] = parsePrice(infoBox.find("th", text = 'High alch').parent.td.text)
        info['lowAlch'] = parsePrice(infoBox.find("th", text='Low alch').parent.td.text)
    try:
        info['weight'] = float(infoBox.find("th", text = 'Weight').parent.td.text.replace('kg',''))
    except ValueError:
        pass
    return info

def storeItemCategoriesPickle():
    with open('itemURLs.csv', 'r') as f:
        lines = f.readlines()
        items = [i.split(',')[0] for i in lines]
    itemCats = {}
    i = 0
    for item in items:
        r = requests.get(URL + '/w/' + item)
        soup = BeautifulSoup(r.text, features="html.parser")
        itemCats[item] = [c.text for c in soup.find('div',id='catlinks').find_all('a') if 'href' in c.attrs and 'Category' in c['href']]
        i += 1
        print('{}/{} ({})'.format(i + 1, len(items), (float(i + 1) / float(len(items))) * 100))
    with open('itemCats.pickle', 'wb') as f:
        pickle.dump(itemCats,f)


def storeItemInfoTSV():
    with open('itemURLs.csv', 'r') as f:
        lines = f.readlines()
        items = [i.split(',')[0] for i in lines]
    start = 0
    try:
        with open('itemsInfo.tsv','r') as f:
            lines = f.readlines()
            if len(lines)>0:
                name = lines[-1].split('\t')[0]
                start =  items.index(name)+1
    except FileNotFoundError:
        pass
    browser = webdriver.Chrome('./chromedriver')
    with open('itemsInfo.tsv','a') as f:
        for i in range(start,len(items)):
            variants = getVariants(items[i])
            if len(variants) == 1:
                info = parseItemInfo(variants[0], browser)
                for key in KEYS:
                    f.write(str(info[key]) + '\t')
                f.write('\n')
            else:
                for v in getVariants(items[i]):
                    browser = webdriver.Chrome('./chromedriver')
                    info = parseItemInfo(v,browser)
                    for key in KEYS:
                        f.write(str(info[key])+'\t')
                    f.write('\n')
            print('{}/{} ({})'.format(i+1,len(items),(float(i+1)/float(len(items)))*100))

def TSVtoPickle():
    with open('itemsInfo.tsv', 'r') as f:
        lines = f.readlines()
    items = [a.split('\t') for a in lines]
    arr = []
    for item in items:
        dic = {}
        i = 0
        for k in KEYS:
            dic[k] = item[i]
            i+=1
        arr.append(dic)
    with open('itemInfo.pickle', 'wb') as f:
        pickle.dump(arr,f)

def getHistoricalPrices(item):
    item = item.replace('+', '%2B').replace(' ','_')
    r = requests.get('{}/w/Module:Exchange/{}/Data?action=raw'.format(URL, item))
    if r.status_code!=200:
        return None
    p = ast.literal_eval(r.text.replace('return {','[').replace('}',']').replace('\n',''))
    for i in range(len(p)):
        a = p[i].split(':')
        if len(a)==2:
            p[i] = (int(a[0]),int(a[1]),-1)
        elif len(a)==3:
            p[i] = (int(a[0]),int(a[1]),float(a[2]))
        else:
            print('Parsing error')
    #pprint(p)
    return p

def storeItemPricesPickle():
    with open('itemInfo.pickle', 'rb') as f:
        itemInfo = pickle.load(f)

    info = {}
    i = 0
    for item in itemInfo:
        prices = getHistoricalPrices(item['name'])
        if prices!=None:
            info[item['name']] = prices
        print('{}/{} ({})'.format(i + 1, len(itemInfo), (float(i + 1) / float(len(itemInfo))) * 100))
        i+=1

    with open('itemPrices.pickle', 'wb') as f:
        pickle.dump(info,f)

def loadItemInfo():
    with open('itemInfo.pickle','rb') as f:
        return pickle.load(f)

def loadItemPrices():
    with open('itemPrices.pickle','rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    storeItemCategoriesPickle()