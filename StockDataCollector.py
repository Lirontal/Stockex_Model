import bulbea as bb
from pprint import pprint as pp
import json
import quandl
from datetime import datetime, timedelta
import pandas_datareader.data as web
import sys
quandl.ApiConfig.api_key = "3JsqccnL2kPvfJ2ekA-a"


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

class StockHandler:
    def __init__(self, stockInfoProvider):
        self.sip = stockInfoProvider
        self.lastUpdated = ""
    def __openStock(self, symbol, source = 'WIKI'):
        self.stock = bb.Share(source = 'WIKI', ticker = symbol)

    def __quandl(self, symbol, start, end):
        data = quandl.get("WIKI/" + symbol, start_date=start, end_date=end,returns="pandas")
        del data['Ex-Dividend'], data['Split Ratio']
        data = data.transform(lambda row: row[:10], axis=1) #Clean timestamps
        return data

    def getStockInfoForDay(self, symbols, date):
        return self.getStockInfoHistory(symbols, date, date)

    def getStockInfoHistory(self, symbols, s_date, e_date):
        start_date = datetime.strptime(s_date,"%Y-%m-%d")-timedelta(1)
        end_date = datetime.strptime(e_date, "%Y-%m-%d")-timedelta(1)
        data = {}
        if (len(symbols) == 0): return data
        with HiddenPrints():
            data = web.DataReader(symbols, 'iex', start=start_date,
                                  end=end_date)  # quandl.get("XNAS/" + symbol+"_UADJ", start_date=date, end_date=date, returns="pandas")
            while (len(data) == 0):
                start_date -= timedelta(1)
                end_date -= timedelta(1)
                data = web.DataReader(symbols, 'iex', start=start_date, end=end_date)

        to_rem = set()
        for key, dataframe in data.items():
            company_info = self.sip.getStockInfo(key)
            if (company_info["symbol"] != company_info["symbol"]):
                company_info["symbol"] = "N/A"
            dataframe["symbol"] = company_info["symbol"] #TODO: ADD CORRECTLY
            if (company_info["company_name"] != company_info["company_name"]):
                company_info["company_name"] = "N/A"
            dataframe["company_name"] = company_info["company_name"]
            if (company_info["company_sector"] != company_info["company_sector"]):
                company_info["company_sector"] = "N/A"
            dataframe["company_sector"] = company_info["company_sector"]

            if(len(dataframe) == 0):
                to_rem.add(key)
                break
            for d in dataframe.iloc[0]:
                if d != d:
                    to_rem.add(key)
                    break
        for k in to_rem:
            data.pop(k)
        self.lastUpdated = start_date.strftime("%Y-%m-%d")
        return data

    def getHistorical(self, symbol, start, end):
        data = self.getStockInfoHistory(symbol, start, end)
        return self.__clean_timestamps(json.loads(data.to_json(orient='index',date_format='iso')))

    def __bulbea(self, symbol, start, end):
        self.__openStock(symbol)
        data = self.stock.data
        data = data[(data.index <= end) & (data.index >= start)]
        return data

    def getLastActiveDay(self,date):
        date = datetime.strptime(date,"%Y-%m-%d") - timedelta(1)
        print(date.weekday())
        while(5 >= date.weekday() >= 6):
            date -= timedelta(1)
        return date

    def __clean_timestamps(self, jsonObj):
        for datetime in list(jsonObj):
            jsonObj[datetime[:10]] = jsonObj.pop(datetime)
        return jsonObj

    def test(self):
        print("Testing the handler with Google's stock, 2017-01-01 through 2017-01-06...")
        self.__openStock('GOOG')
        z = self.getHistorical('GOOG','2017-01-01', '2017-01-06')
        pp(z)