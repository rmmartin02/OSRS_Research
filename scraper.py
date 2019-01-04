from bs4 import BeautifulSoup
import requests

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

def getItemInfo(soup):
    infoBox = soup.find('tbody').find_all('tr')
    for info in infoBox:
        print(info.text)
    info = {}

    info['name'] = infoBox[0].text
    info['released'] = infoBox[2].text[8:]
    info['update'] = 'N/A'

    if 'Update' in infoBox[2].text[8:]:
        info['released'] = info['released'].replace(' (Update)','')
        info['update'] = infoBox[2].find_all('a')[-1]['href']

    info['members'] = infoBox[3].text[7:]
    info['questItem'] = infoBox[4].text[10:]
    info['tradeable'] = infoBox[6].text[9:]
    info['equipable'] = infoBox[7].text[9:]
    info['stackable'] = infoBox[8].text[9:]
    info['destroy'] = infoBox[9].text[7:]
    info['highAlch'] = infoBox[11].text[9:-6].replace(',','')
    info['lowAlch'] = infoBox[12].text[8:-6].replace(',','')
    info['storePrice'] = infoBox[13].text[11:].replace(',','')
    info['exchangePrice'] = infoBox[14].text[8:-13].replace(',','')
    info['buyLimit'] = infoBox[15].text[9:].replace(',','')
    info['weight'] = infoBox[16].text[6:-3]
    info['examine'] = infoBox[18].text
    info['categories'] = [c.text for c in soup.find(id='catlinks').find_all('a') if 'href' in c.attrs and 'Category' in c['href']]
    return info

def getExchangePrices(item):
    r = requests.get('{}/w/Module:Exchange/{}/Data?action=raw'.format(URL,item))
    print(r.text[7:])

if __name__ == "__main__":
    #getItemInfo('/w/Abyssal_whip')
    #itemURL = '/w/Abyssal_whip'
    #r = requests.get(URL + itemURL)
    #soup = BeautifulSoup(r.text, features="html.parser")
    #print(getItemInfo(soup))
    getExchangePrices('Abyssal_whip')