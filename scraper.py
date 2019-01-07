from bs4 import BeautifulSoup
import requests
import re

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

def parseItemInfo(item):
    print(item)
    r = requests.get(URL + '/w/' + item)
    soup = BeautifulSoup(r.text, features="html.parser")
    infoBox = soup.find_all('div',{'class':'infobox-wrapper'})[0]
    print(r.text)
    info = {}
    info['name'] =  infoBox.find_all("th",{'data-attr-param':'name'})[0].text
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
    print(infoBox.find_all("a", text="info")[0])
    print(info)
    return info

def storeItemInfo():
    items = []
    with open('itemURLs.csv', 'r') as f:
        lines = f.readlines()
        items = [i.split(',')[0] for i in lines]
    with open('itemsInfo.csv','w') as f:
        i = 0
        for item in items:
            item = 'Dragon_dagger'
            variants = getItemInfo(item)
            for info in variants:
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
                i+=1
                print('{}/{} ({})'.format(i,len(items),float(i)/float(len(items))))

def getExchangeInfo(item):
    '''return {
    itemId     = 11095,
    price      = 2263,
    last       = 2254,
    date       = '06 January 2019 00:00:00 (UTC)',
    lastDate   = '05 January 2019 00:00:00 (UTC)',
    icon       = 'Abyssal bracelet.png',
    item       = 'Abyssal bracelet',
    value      = 4200,
    limit      = 10000,
    members    = true,
    category   = nil,
    examine    = 'Makes using the Abyss just slightly safer.',
    hialch     = 2520,
    lowalch    = 1260
    }'''
    r = requests.get('{}/w/Module:Exchange/{}?action=raw'.format(URL, item))
    info = {}
    if len(r.text) == 0:
        info['price'] = -1
        info['limit'] = -1
        return info
    print(r.text)
    arr = r.text[8:-1].split(',')
    for a in arr:
        if 'price' in a:
            info['price'] = int(a.split('=')[1])
        if 'limit' in a:
            info['limit'] = int(a.split('=')[1])
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
    #print(getExchangeInfo('Abyssal_dagger'))