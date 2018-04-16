import pandas
import json
import stockDataCollector
class Algorithm:
    def __init__(self):
        self.sdc = stockDataCollector()

    def getEasySearch(self, budget):
        b = int(budget)
        #TODO: acquire all stock symbols
        symbols = ['GOOGL', 'AAPL']
        for s in symbols:

        data = pandas.DataFrame(index=['GOOGL', 'AAPL'], data={})
        return json.loads(data).to_json()

    def getAdvSearch(self, budget, company_type, company_name):
        return json.loads(pandas.DataFrame(index=['GOOGL', 'AAPL'], data={'Score': [57, 33]}).to_json())

    def getRecommend(self):
        #TODO: check mongoDB for previous recommendations and return if they are recent
        companies = ['GOOGL', 'AAPL']
        data = {'Score': '[57, 33]'}
        return pandas.DataFrame(index=companies, data=data).to_json()


    def getScoreForStock(self, symbol):
        if(symbol=='GOOGL'):
            return 57
        else: return 33
