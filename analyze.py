import scraper
import matplotlib.pyplot as plt

def priceChanges(prices):
    absChange = [0]*(len(prices)-1)
    percChange = [0]*(len(prices)-1)
    for i in range(0,len(prices)-1):
        absChange[i] = prices[i+1]-prices[i]
        percChange[i] = absChange[i]/prices[i]
    return absChange,percChange

def marginChanges(prices,buylimit):
    absChange = [0]*(len(prices)-1)
    percChange = [0]*(len(prices)-1)
    for i in range(0,len(prices)-1):
        absChange[i] = prices[i+1]*buylimit-prices[i]*buylimit
        percChange[i] = absChange[i]/(prices[i]*buylimit)
    return absChange,percChange

if __name__ == "__main__":
    a = scraper.getExchangePrices('Abyssal_whip')
    info = scraper.getItemInfo('Abyssal_whip')
    b = priceChanges(a[1])
    plt.plot(a[0][1:],b[0])
    plt.show()
    plt.plot(a[0],a[1])
    plt.show()