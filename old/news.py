from bs4 import BeautifulSoup
import datetime
from datetime import date, timedelta
import pickle # object for serialization
import requests

FNAME = "snp500_formatted"
stocks = []

def getNewsForDate(date):
    file = open('data/news/' + date.strftime('%Y-%m-%d') + '.csv', 'w')
    print('Getting news for ' + date.strftime('%Y-%m-%d'))
    for i in range(len(stocks)):
        query = 'http://www.reuters.com/finance/stocks/companyNews?symbol=' + stocks[i] + '&date=' + format(date.month, '02d') + format(date.day, '02d') + str(date.year)
        print('Getting news for ' + stocks[i])

        response = requests.get(query)
        soup = BeautifulSoup(response.text, "html.parser")
        divs = soup.findAll('div', {'class': 'feature'})
        print('Found ' + str(len(divs)) + ' articles.')

        if(len(divs) == 0):
            continue

        data = u''
        i=0
        for div in divs:
            if(i<10):
                u = div.findAll(text=True)
                print(str(type(u)))
                data = data.join(u)
                i+=1
            else: break

        x = data.replace('\n', ' ')
        #z=data.encode("utf-8").replace('\n', ' ')
        print("x:"+str(type(x)))
        z = stocks[i] + ',' + x

        file.write(z)
        file.write('\n')
    file.close()

def getNews():
    # dataHistFile = open('dat.pkl', 'rb')
    # dataHist = pickle.load(dataHistFile)
    # date = dataHist['last_updated'] + datetime.timedelta(days=1)
    endDate = datetime.date.today()
    date = datetime.date.today()
    print(date)
    while(date <= endDate):
        getNewsForDate(date)
        date += datetime.timedelta(days=1)

    # dataHist['last_updated'] = endDate
    # dataHistFile.seek(0)
    # pickle.dump(dataHist, dataHistFile, protocol = pickle.HIGHEST_PROTOCOL)
    # dataHistFile.close()

def init():
    global stocks
    with open(FNAME,'r') as f:
        stocks = f.readlines()
    for i in range(len(stocks)):
        stocks[i] = stocks[i].rstrip('\n')
    getNews()
init()
#getNewsForDate(date.today()-timedelta(weeks = 1, days=4))