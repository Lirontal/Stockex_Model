import pandas
import json
import StockDataCollector
from datetime import datetime, timedelta
from StockInfoProvider import StockInfoProvider
from StockDataCollector import StockHandler
from pprint import pprint as pp
import pandas_datareader.data as web
from numpy import array
class Algorithm:
    def __init__(self):
        self.sip = StockInfoProvider()
        self.sdc = StockHandler(self.sip)
        self.preparedCompanyData = {}
        self.prepareDataOvernight()

    def prepareDataOvernight(self):
        a = self.sip.getAllStocks()
        print("Preparing "+str(len(a))+" stock data entries...")

        i = 3319 #TODO:CHANGE TO 0
        chunk_size = 20 #TODO: CHANGE TO 500
        while(i < len(a)):
            self.preparedCompanyData.update(self.sdc.getStockInfoForDay(list(a[i:i+chunk_size-1]), "2017-04-25"))
            i += chunk_size
        print("Done preparing.")

    def getEasySearch(self, budget):
        b = int(budget)
        for index, value in self.preparedCompanyData.items():
            value["score"] = self.getScoreForStock(index, budget)
        return self.preparedCompanyData

    def getAdvSearch(self, budget, company_type, company_name):
        return self.easySearch(budget)

    def getRecommend(self,symbol):
        #TODO: check DB for previous recommendations and return if they are recent
        return 10.0

    def getScoreForStock(self, symbol, budget):
        if symbol == 'GOOG':
            return 100*budget
        else:
            return 1*budget
