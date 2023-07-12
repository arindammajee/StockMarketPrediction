import os
from helper.stockCollection import stockCollection
from helper.newsCollection import newsCollection
#from helper.lstm import LSTMModel
from helper.timeEmbeddingTransformer import *
from helper.dataprocess import DataProcess
from datetime import date
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def StockPrediction(comp):
   # Create a directory for that company
   dirPath = os.getcwd() + '/companies'
   if os.path.exists(dirPath)==False:
      os.mkdir(dirPath)
      
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
   todayStockPrice = None
   tomorrowStockPrice = None

   if stockObj._date==None:
      print("We haven't tracked {}'s share prices before. Will you allow me some time to update myself?".format(stockObj.stock))
      dateFlag = input('Enter 1 if yes else 0: ')

      if(dateFlag=='1'):
         stockObj.historicalDataCollection()
         print("Collected the Stock Data Successfully!")
         newsObj.collectGoogleNews([comp])
         print("Collected the news Successfully!")
         
         log_path = os.path.join(compDir, 'log.txt')
         try:
            with open(log_path, 'w') as logObject:
               logObject.write(str(date.today()))
               logObject.close()
               stockFLAG = True
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
         newsObj.collectGoogleNews([comp])
         print("Collected the news Successfully!")
         log_path = os.path.join(compDir, 'log.txt')
         try:
            with open(log_path, 'w') as logObject:
               logObject.write(str(date.today()))
               logObject.close()
               stockFLAG = True
               modelPath = os.path.join(modelDir, 'Transformer.h5')
         except:
            print("Failed to update log file")
      else:
         print('Something wrong happend!')
      
   elif stockObj._date==str(date.today()):
      stockFLAG = True
      modelPath = os.path.join(modelDir, 'Transformer.h5')
      if os.path.exists(modelPath):
         trainedSystem = True
         print("I have updated information and model with me. Just wait few seconds to get tommorow's prediction.")



   if stockFLAG==True:
      # Process the Data
      WindowSize=32
      preProcessedData = DataProcess(file_path=stockObj.stockFilePath, columnns=['Open'])
      preProcessedData.split(window=WindowSize, ratio=[0.95, 0.05])
      x_train = preProcessedData.x_train
      x_val = preProcessedData.x_test
      y_train = preProcessedData.y_train
      y_val = preProcessedData.y_test

      if trainedSystem==False:
         # Build The Model
         modelObj = TimeEmbeddedTransformerModel()
         modelObj.create_model()
         modelObj.sc = preProcessedData.sc

         # Fit and Evaluate
         modelObj.fitModel((x_train, y_train), val_data=(x_val, y_val))
         modelObj.train_loss = modelObj.evaluateModel((x_train, y_train))
         modelObj.val_loss = modelObj.evaluateModel((x_val, y_val))
         modelObj.train_mse = modelObj.evaluateMSE((x_train, y_train))
         modelObj.val_mse = modelObj.evaluateMSE((x_val, y_val))

         # Save the Model History
         # convert the history.history dict to a pandas DataFrame:
         with open(os.path.join(modelDir, 'history.csv'), mode='w') as f:
            pd.DataFrame(modelObj.history.history).to_csv(f)
         
         with open(os.path.join(modelDir, 'performance.txt'), mode='w') as f:
            f.write("Model Train Loss: " + str(modelObj.train_loss) + "\n")
            f.write("Model Validation Loss: " + str(modelObj.val_loss) + "\n")
            f.write("Model Train MSE: " + str(modelObj.train_mse) + "\n")
            f.write("Model Validation MSE: " + str(modelObj.val_mse) + "\n")
         
         modelObj.model.save(modelPath)

         # Plot the Data
         actual = np.concatenate((np.reshape(y_train, newshape=(y_train.shape[0], 1)), np.reshape(y_val, newshape=(y_val.shape[0], 1))), axis=0)
         actualscaled = preProcessedData.sc.inverse_transform(actual)
         modelpredicted = modelObj.predictModel(np.concatenate((x_train, x_val), axis=0))
         plt.figure()
         plt.plot(actualscaled, color = 'red', label = 'Real price')
         plt.plot(modelpredicted, color = 'blue', label = 'Predicted price on Validation data')
         plt.plot(modelpredicted[:x_train.shape[0]], color = 'green', label = 'Predicted price on Train data')
         plt.savefig(os.path.join(modelDir, "TrainingFigure.png"))
         plt.show()

         # Prediction of Tomorrow's Stock
         nextTestData = x_val[-1]
         todayStock = y_val[-1]
         nextTestData = np.append(nextTestData[1:, :], [[todayStock]], axis=0)
         nextTestData = np.reshape(nextTestData, newshape=(1, nextTestData.shape[0], nextTestData.shape[1]))

         predict = modelObj.model.predict(nextTestData)
         todayStockPrice = preProcessedData.sc.inverse_transform([[todayStock]])[0][0]
         tomorrowStockPrice = preProcessedData.sc.inverse_transform(predict)[0][0]
         print("Today {} stock price is {} and our system says according to the trend tomorrow it will be {}".format(comp, str(todayStockPrice), str(tomorrowStockPrice)))
         print(nextTestData)
      else:
         # Prediction of Tomorrow's Stock
         nextTestData = x_val[-1]
         todayStock = y_val[-1]
         nextTestData = np.append(nextTestData[1:, :], [[todayStock]], axis=0)
         nextTestData = np.reshape(nextTestData, newshape=(1, nextTestData.shape[0], nextTestData.shape[1]))

         modelObj = TimeEmbeddedTransformerModel()
         modelObj.model = modelObj.LoadModel(modelPath)
         predict = modelObj.model.predict(nextTestData)
         todayStockPrice = preProcessedData.sc.inverse_transform([[todayStock]])[0][0]
         tomorrowStockPrice = preProcessedData.sc.inverse_transform(predict)[0][0]
         print("Today {} stock price is {} and our system says according to the trend tomorrow it will be {}".format(comp, str(todayStockPrice), str(tomorrowStockPrice)))
         print(nextTestData)
   else:
      print("Something wrong happend with the System!")

   return todayStockPrice, tomorrowStockPrice

if __name__=="__main__":
   html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Stock Market Prediction </h2>
    <h5> Welcome to our AI - based Stock Market Prediction System. </h5>
    </div>
    """
   st.markdown(html_temp, unsafe_allow_html = True)
   if(True):
      comp = st.text_input("Enter the company name: ", "Write Here", key="comp")
      if st.button("Predict Next Day's Stock Value"):
         todayStockPrice, tomorrowStockPrice = StockPrediction(comp)
         st.success("Today {} stock price is {} and our system says according to the trend tomorrow it will be {}".format(comp, str(todayStockPrice), str(tomorrowStockPrice)))