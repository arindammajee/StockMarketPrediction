import os
from helper.stockCollection import stockCollection
from helper.newsCollection import newsCollection
from helper.lstm import LSTMModel
from helper.dataprocess import DataProcess
from helper.newsSentiment import SentimentAnalysis
from helper.TickerSymbol import TickerSymbol
from datetime import date, timedelta, datetime
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import re

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def isNowInTimePeriod(startTime, endTime, nowTime):
    if startTime < endTime:
        return nowTime >= startTime and nowTime <= endTime
    else: #Over midnight
        return nowTime >= startTime or nowTime <= endTime


timeStart = '12:00AM'
timeEnd = '3:30PM'
timeNow = datetime.now().strftime("%I:%M%p")
timeEnd = datetime.strptime(timeEnd, "%I:%M%p")
timeStart = datetime.strptime(timeStart, "%I:%M%p")
timeNow = datetime.strptime(timeNow, "%I:%M%p")
timeFlag = isNowInTimePeriod(timeStart, timeEnd, timeNow)

def StockPrediction(company):
   # Create a directory for that company
   dirPath = os.getcwd() + '/companies'
   if os.path.exists(dirPath)==False:
      os.mkdir(dirPath)
   
   comp = TickerSymbol.get_symbol(company)
   compDir = os.path.join(dirPath, comp)
   if os.path.exists(compDir)==False:
      os.mkdir(compDir)
      
      
   # Collect the stock data and news for that company
   dataDir = os.path.join(compDir, 'Data')
   if os.path.exists(dataDir)==False:
      os.mkdir(dataDir)

   # Model Directory
   modelDir = os.path.join(compDir, 'Model')
   if os.path.exists(modelDir)==False:
      os.mkdir(modelDir)

   stockObj = stockCollection(comp)
   newsObj = newsCollection(comp)
   stockFLAG = False
   trainedSystem = False
   isNewData = False
   todayStockPrice = None
   modelPath = os.path.join(modelDir, 'LSTM.h5')

   if stockObj._date==None:
      print("We haven't tracked {}'s share prices before. Will you allow me some time to update myself?".format(stockObj.stock))
      dateFlag = input('Enter 1 if yes else 0: ')

      if(dateFlag=='1'):
         stockObj.historicalDataCollection()
         print("Collected the Stock Data Successfully!")
         newsObj.collectGoogleNews([company])
         print("Collected the news Successfully!")
         
         log_path = os.path.join(compDir, 'log.txt')
         try:
            with open(log_path, 'w') as logObject:
               logObject.write(str(date.today()))
               logObject.close()
               stockFLAG = True
               isNewData = True
         except:
            print("Failed to update log file")
      else:
         print('Something wrong happend!')

   elif stockObj._date!=str(date.today()):
      print("Today is {} but I am updated till {}'s market information. Will you allow me some time to update myself or you tomorrow's prediction based on information what I have?".format(date.today(), stockObj._date))
      dateFlag = input('Enter 1 if you can allow me to update myself else 0: ')

      if(dateFlag=='1'):
         stockObj.historicalDataCollection()
         print("Collected the updated stock data Successfully!")
         newsObj.collectGoogleNews([company])
         print("Collected the news Successfully!")
         log_path = os.path.join(compDir, 'log.txt')
         try:
            with open(log_path, 'w') as logObject:
               logObject.write(str(date.today()))
               logObject.close()
               stockFLAG = True
               isNewData = True
               if os.path.exists(modelPath):
                  trainedSystem = True
               
         except:
            print("Failed to update log file")
      else:
         print('Something wrong happend!')
      
   elif stockObj._date==str(date.today()):
      stockFLAG = True
      newsObj.collectGoogleNews([company])
      print("Collected the news Successfully!")
      if os.path.exists(modelPath):
         trainedSystem = True
         print("I have updated information and model with me. Just wait few seconds to get tommorow's prediction.")



   if stockFLAG==True:
      # Process the Data
      WindowSize=32
      n_future = 2
      features = ['Open', 'High', 'Low', 'Close']
      target = ['Open']
      preProcessedData = DataProcess(file_path=stockObj.stockFilePath, window=WindowSize, n_future=n_future, columnns=features)
      preProcessedData.split(ratio=[0.8, 0.1, 0.1])
      x_train = preProcessedData.x_train
      x_val = preProcessedData.x_val
      x_test = preProcessedData.x_test
      y_train = preProcessedData.y_train
      y_val = preProcessedData.y_val
      y_test = preProcessedData.y_test
      todayStock = preProcessedData.last[-1]
      #print(todayStock)
      #print(todayStock.shape)

      if isNewData==True:
         if trainedSystem == False:
            # Build The Model
            modelObj = LSTMModel(n_future, len(target), modelPath)
            modelObj.buildModel(input_shape = x_train.shape[1:])
            modelObj.sc = preProcessedData.sc
            # Fit and Evaluate
            modelObj.fitModel((x_train, y_train), val_data=(x_val, y_val))
         elif trainedSystem == True:
            modelObj = LSTMModel(n_future, len(target), modelPath)
            modelObj.model = modelObj.LoadModel(modelPath)
            print("We have a trained model so with the new data we will do fine tuning.\n")
            modelObj.sc = preProcessedData.sc
            # Fit and Evaluate
            modelObj.fineTuneModel((x_train, y_train), val_data=(x_val, y_val))
         
         modelObj.train_loss = modelObj.evaluateModel((x_train, y_train))
         modelObj.val_loss = modelObj.evaluateModel((x_val, y_val))
         modelObj.test_loss = modelObj.evaluateModel((x_test, y_test))
       


         # Save the Model History
         # convert the history.history dict to a pandas DataFrame:
         with open(os.path.join(modelDir, 'history.csv'), mode='w') as f:
            pd.DataFrame(modelObj.history.history).to_csv(f)
         
         with open(os.path.join(modelDir, 'performance.txt'), mode='w') as f:
            f.write("Loss Parameters: MSE, MAPE, MAE")
            f.write("Model Train Loss: " + str(modelObj.train_loss))
            f.write("\nModel Validation Loss: " + str(modelObj.val_loss))
            f.write("\nModel test Loss: " + str(modelObj.test_loss))
            #f.write("\nModel Train MSE: " + str(modelObj.train_mse))
            #f.write("\nModel Validation MSE: " + str(modelObj.val_mse))
            #f.write("\nModel Test MSE: " + str(modelObj.test_mse))
         
         modelObj.model.save(modelPath)

         # Plot the Data
         actualOpenData = preProcessedData.data
         allScaledInput = np.concatenate((x_train, x_val), axis=0)
         allScaledInput = np.concatenate((allScaledInput, x_test), axis=0)
         modelpredicted = modelObj.model.predict(allScaledInput)
         modelpredicted = modelpredicted[:,0]
         modelpredicted = np.concatenate((modelpredicted, actualOpenData[:modelpredicted.shape[0], 1:]), axis=1)
         predictedRescaled = preProcessedData.sc.inverse_transform(modelpredicted)
         predictedRescaled = np.concatenate((np.zeros((33, 4)), predictedRescaled), axis=0)
         plt.figure() 
         plt.plot(actualOpenData[:, 0], color = 'red', label = 'Real price')
         plt.plot(predictedRescaled[:, 0], color = 'yellow', label = 'Predicted price on Test data')
         plt.plot(predictedRescaled[:x_train.shape[0]+x_val.shape[0], 0], color = 'blue', label = 'Predicted price on Validation data')
         plt.plot(predictedRescaled[:x_train.shape[0], 0], color = 'green', label = 'Predicted price on Train data')
         plt.legend(["Real", "Test", "Validation", "Train"])
         plt.xlabel('No of Days')
         plt.ylabel("Openning stock value")
         plt.savefig(os.path.join(modelDir, "TrainingFigure.png"), dpi=900)
         plt.show()


      else:
         # Prediction of Tomorrow's Stock
         modelObj = LSTMModel(n_future, len(target), modelPath)
         modelObj.model = modelObj.LoadModel(modelPath)
         modelObj.sc = preProcessedData.sc


      nextDaysStock = modelObj.model.predict(np.reshape(todayStock, newshape=(1, todayStock.shape[0], todayStock.shape[1])))
      nextDaysStock = np.reshape(nextDaysStock, newshape=(nextDaysStock.shape[1], nextDaysStock.shape[2]))
      nextDaysStock = np.concatenate((nextDaysStock, todayStock[:nextDaysStock.shape[0], 1:]), axis=1)

      todayStockPrice = preProcessedData.sc.inverse_transform(todayStock)
      nextDaysStocPrice = preProcessedData.sc.inverse_transform(nextDaysStock)

      todayDate = date.today()
      if str(todayDate)==preProcessedData.last_date:
         printStr = 'Today is '
         printStr = printStr + str(todayDate) + ' and the openning stock price of ' + company + '('+ comp + ') is: ' + str(todayStockPrice[-1][0]) + '.\nAnd our AI system says that\n'
         for i in range(nextDaysStocPrice.shape[0]):
            printStr = printStr + 'On ' + str(todayDate + timedelta(days=i+1)) + ' openning Stock Price Will be: ' + str(nextDaysStocPrice[i][0]) + '\n'
      else:
         printStr = 'Today is '
         printStr = printStr + str(todayDate) + ' and on ' + preProcessedData.last_date + ' the openning stock price of ' + company + '('+ comp + ') was: ' + str(todayStockPrice[-1][0]) + '.\nAnd our AI system says that\n'
         for i in range(nextDaysStocPrice.shape[0]):
            printStr = printStr + 'On ' + str(todayDate + timedelta(days=i+1)) + ' openning Stock Price will be: ' + str(nextDaysStocPrice[i][0]) + '\n'

      print(printStr)

      # News Sentiment Analysis
      newsData = SentimentAnalysis(newsObj.filepath)
      newsData.cleanedData()
      sentiment = newsData.sentiment_scores()


      return printStr, sentiment
         
   else:
      print("Something wrong happend with the System!")

   return 0, 0

if __name__=="__main__":
   html_temp = """
    <div style="background:#8080FF; padding:10px; allign:center;">
    <h1 style="color:white;text-align:center;">Stock Market Prediction </h1>
    <h4 style="text-align:center;"> Welcome to our AI - based Stock Market Prediction System. </h4>
    </div>
    """
   st.components.v1.html(html_temp)
   comp = st.text_input("Enter the company name: ", "Write Here", key="comp")
   if st.button("Predict Next Day's Stock Value"):
      prtStr, sentiment = StockPrediction(comp)
      #prtStr = re.sub(r'\n', '<br>', prtStr)
      st.success(prtStr)
      st.success("And based on last 24 hour news our sentiment analyser says that tomorrow {} stock price will {}".format(comp, sentiment))
      