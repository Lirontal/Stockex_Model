import pandas
import json
import stockDataCollector
from datetime import datetime, timedelta
from StockInfoProvider import StockInfoProvider
from stockDataCollector import StockHandler
from pprint import pprint as pp
import pandas_datareader.data as web
from numpy import array
class Algorithm:
    def __init__(self):
        self.sip = StockInfoProvider()
        self.sdc = StockHandler(self.sip)
        self.preparedCompanyData = {}
    def prepareDataOvernight(self):
        #TODO: REQUEST MULTIPLE SYMBOLS FOR QUICK ANSWER
        a = self.sip.getAllStocks()
        # a = self.sip.getAllStocks()
        pp('Got '+str(len(a))+' stocks')
        # pp(a[3000:3499])
        # for symbol in :
        # pp(type(a))
        # pp(self.sdc.getStockInfoForDay("Y", "2017-04-25"))
        i = 3319#TODO:CHANGE TO 0
        chunk_size = 20
        while(i < len(a)):
            # pp("i: " + str(i) + ", end: " + str(i + chunk_size - 1))
            self.preparedCompanyData.update(self.sdc.getStockInfoForDay(list(a[i:i+chunk_size-1]), "2017-04-25"))
            i += chunk_size

        # z = a[3000:3338]
        # pp(z)
        # q = (self.sdc.getStockInfoForDay(list(a[3000:3338]), "2017-04-25"))
        # pp(q)
        # pp(self.preparedCompanyData["Y"])

    def getEasySearch(self, budget):
        b = int(budget)
        #TODO: acquire all stock symbols - remove placeholder
        # symbols = ['GOOGL', 'AAPL']
        # columns = ['Symbol','Score','Open','High','Low','Close','Volume','Adj_Open','Adj_High','Adj_Low','Adj_Close','Adj_Volume']
        # data = pandas.DataFrame([['GOOGL',2*b,3,4,5,6,7,8,9,10,11,12]],columns=columns)
        # newRow = pandas.DataFrame([['AAPL',3*b,3,4,5,6,7,8,9,10,11,12]],columns=columns)
        # print(self.getLastActiveDay())
        self.prepareDataOvernight()
        for index,value in self.preparedCompanyData.items():
            value["score"] = self.getRecommend(index)
        # pp((self.preparedCompanyData))
        pp(type(self.preparedCompanyData))
        return self.preparedCompanyData
        # retData = data.append(newRow,True)
        # return retData.to_json(orient='index')

    def getAdvSearch(self, budget, company_type, company_name):
        return self.easySearch(budget)

    def getRecommend(self,symbol):
        #TODO: check mongoDB for previous recommendations and return if they are recent
        return 10.0


    def getScoreForStock(self, symbol):
        if(symbol=='GOOGL'):
            return 57
        else: return 33

# a = Algorithm()
# (a.getEasySearch(100))
