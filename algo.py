from datetime import datetime, timedelta
from StockInfoProvider import StockInfoProvider
from StockDataCollector import StockHandler
from LSTM import StockModel
from urllib import error as urllib_err


class Algorithm:
    def __init__(self):
        self.sip = StockInfoProvider()
        self.sdc = StockHandler(self.sip)
        self.preparedCompanyData = {}

        self.sm = StockModel()

        self.start_date = "2018-05-20"
        self.end_date = "2018-05-23"

        self.prepareDataOvernight()
        self.lastUpdated = ""


    def prepareDataOvernight(self):
        a = self.sip.getAllStocks()
        print("Preparing "+str(len(a))+" stock data entries...")

        i = 0 #TODO:CHANGE TO 0
        chunk_size = 20 #TODO: CHANGE TO 700
        while i < 5:#i < len(a):
            entry = self.sdc.getStockInfoForDay(list(a[i:i+chunk_size-1]), datetime.today())#"2017-04-25")#datetime.today().strftime('%Y-%m-%d'))''''list(a[i:i+chunk_size-1])'''
            self.lastUpdated = self.sdc.lastUpdated
            self.preparedCompanyData.update(entry)
            i += chunk_size

        # print(self.preparedCompanyData["AA"]) #TEST
        # Calc score for each stock
        latest = datetime.today() - timedelta(4)
        for index, value in self.preparedCompanyData.items():
            try:
                self.sm.start(index,self.start_date,self.end_date, latest, latest)
            except (urllib_err.HTTPError,TypeError):
                continue
            print(index)
            value["score"] = self.getScoreForStock(index)
        # print("COMP DATA "+str(self.preparedCompanyData))
        print("Done preparing.")

    def getEasySearch(self, budget):
        b = int(budget)
        # print(self.preparedCompanyData)
        # pp(self.preparedCompanyData)
        return {k: df for k, df in self.preparedCompanyData.items() if df.head(1)["open"][0] <= b}

    def getAdvSearch(self, budget, company_type, company_name):
        d = self.getEasySearch(budget)
        # print("********** PRINTING *************")

        for key, value in d.items():
            print(str(value))
        # d =
        d = {k: df for k, df in d.items() if ("company_name" in df.head(1))}
        # try:
        return {k: df for k, df in d.items() if  (company_name.lower() in df.head(1)["company_name"][0].lower()) & (company_type.lower() in df.head(1)["company_sector"][0].lower())}
        # except KeyError:
        #     return d

    def getRecommend(self,symbol):
        #TODO: check DB for previous recommendations and return if they are recent
        return 10.0

    def getScoreForStock(self, symbol):
        try:
            return self.sm.stockDataDict[symbol]
        except (KeyError):
            pass


# algo = Algorithm()
# algo.prepareDataOvernight()