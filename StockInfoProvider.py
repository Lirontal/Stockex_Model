import pandas as pd


class StockInfoProvider:
    def __init__(self):
        self.stocks_info = pd.read_csv("stocks_info.csv", index_col="Ticker")

    def getStockInfo(self, symbol):
        infoRow = self.stocks_info.loc[symbol, :]
        if(infoRow.Sector != infoRow.Sector):
            infoRow.Sector="N/A"
        if (infoRow.Name != infoRow.Name):
            infoRow.Name = "N/A"

        return {"symbol": symbol, "company_name": infoRow.Name, "company_sector": infoRow.Sector}

    def getAllStocks(self):
        return self.stocks_info.index.values


