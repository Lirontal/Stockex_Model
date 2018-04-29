import pandas as pd
from pprint import pprint
class StockInfoProvider:
    def __init__(self):
        self.stocks_info = pd.read_csv("stocks_info.csv", index_col="Ticker")

    def getStockInfo(self, symbol):
        # if symbol not in self.stocks_info["Ticker"]:
        #     return {}
        infoRow = self.stocks_info.loc[symbol, :]

        return {"Symbol": symbol, "Company_Name": infoRow.Name, "Company_Sector": infoRow.Sector}

    def getAllStocks(self):
        return self.stocks_info.index.values