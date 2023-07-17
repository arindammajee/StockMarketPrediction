import yfinance as yf
import os
import json
import csv
from datetime import date

# Collect the Historical Stock Data
class stockCollection:
    def __init__(self, stock = 'ADANIPOWER.NS'):
        self.stock = stock
        self.comp_dir = os.path.join(os.path.join(os.getcwd(), 'companies'), self.stock)
        self.data_path = os.path.join(self.comp_dir, 'Data')
        self.log_path = os.path.join(self.comp_dir, 'log.txt')
        self.infoPath = os.path.join(self.data_path, self.stock + '.json')
        self.stockFilePath = os.path.join(self.data_path, self.stock + '.csv')

        # Read the text file
        if os.path.exists(self.log_path):
            logObject = open(self.log_path, 'r')
            lines = logObject.readlines()
            logObject.close()

            # Extract Date
            dateLine = lines[0]
            self._date = dateLine
        
        else:
            self._date = None

    
    def historicalDataCollection(self):
        stockTicker = yf.Ticker(self.stock)
        stockInfo = stockTicker.info
        stocks = stockTicker.history(period="max")
        
        try:
            stocks.to_csv(self.stockFilePath)
        except:
            print("Unable to write the stocks data into csv file")
        

        try:
            with open(self.infoPath, "w") as stockFile:
                json.dump(stockInfo, stockFile)
        except:
            print("Unable to write to file")

    
   
