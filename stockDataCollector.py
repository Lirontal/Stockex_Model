import bulbea as bb
from pprint import pprint as pp
import json
import quandl
quandl.ApiConfig.api_key = "3JsqccnL2kPvfJ2ekA-a"
class StockHandler:
    def __openStock(self, symbol, source = 'WIKI'):
        self.stock = bb.Share(source = 'WIKI', ticker = symbol)


    def __quandl(self, symbol, start, end):
        data = quandl.get("WIKI/" + symbol, start_date=start, end_date=end)
        del data['Ex-Dividend']
        del data['Split Ratio']
        jsonObj = json.loads(data.to_json(orient='index', date_format='iso'))
        self.__clean_timestamps(jsonObj)
        return jsonObj

    def getHistorical(self, symbol, start, end):
        return self.__quandl(symbol, start, end)
        # return self.__bulbea(symbol,start,end)
    def __bulbea(self, symbol, start, end):
        self.__openStock(symbol)
        data = self.stock.data
        data = data[(data.index <= end) & (data.index >= start)]
        jsonObj = json.loads(data.to_json(orient='index', date_format='iso'))
        self.__clean_timestamps(jsonObj)
        return jsonObj

    def __clean_timestamps(self, jsonObj):
        for datetime in list(jsonObj):
            jsonObj[datetime[:10]] = jsonObj.pop(datetime)

    def test(self):
        print("Testing the handler with Google's stock, 2017-01-01 through 2017-01-06...")
        self.__openStock('GOOGL')
        z = self.getHistorical('GOOGL','2017-01-01', '2017-01-06')
        pp(z)

# s = StockHandler()
# s.test()