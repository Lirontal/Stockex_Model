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
        self.lastUpdated = ""

    def prepareDataOvernight(self):
        a = self.sip.getAllStocks()
        print("Preparing "+str(len(a))+" stock data entries...")

        i = 0 #TODO:CHANGE TO 0
        chunk_size = 700 #TODO: CHANGE TO 700
        while(i < len(a)):
            entry = self.sdc.getStockInfoForDay(list(a[i:i+chunk_size-1]), datetime.today().strftime('%Y-%m-%d'))#"2017-04-25")#datetime.today().strftime('%Y-%m-%d'))
            self.lastUpdated = self.sdc.lastUpdated
            self.preparedCompanyData.update(entry)
            i += chunk_size

        # Calc score for each stock
        for index, value in self.preparedCompanyData.items():
            value["score"] = self.getScoreForStock(index)

        print("Done preparing.")
        # pp(self.preparedCompanyData)

    def getEasySearch(self, budget):
        b = int(budget)
        # print(self.preparedCompanyData)

        # pp(self.preparedCompanyData)
        return self.preparedCompanyData

    def getAdvancedSearch(self, budget):
        easyData = self.getEasySearch(self,budget)


    def getAdvSearch(self, budget, company_type, company_name):
        return self.getEasySearch(budget)

    def getRecommend(self,symbol):
        #TODO: check DB for previous recommendations and return if they are recent
        return 10.0

    def getScoreForStock(self, symbol):
        if symbol == 'ZBRA':
            return 100
        else:
            return 87
