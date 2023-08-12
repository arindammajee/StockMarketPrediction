import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from pytz import timezone
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from LSTM_GRU import ShortTermPrediction


class DataDownloader:
    def __init__(self, ticker):
        self.ticker = ticker
        self.period = '1mo'
        self.interval = '5m'
        self.data_cache = {}

    def currentTime(self):
        return datetime.now(timezone('Asia/Kolkata'))
    
    def isMarketOpen(self):
        current_time = self.currentTime()
        if current_time.hour >= 9 and current_time.minute >= 15:
            if current_time.hour <= 15 and current_time.minute <= 30:
                return True
            
        return False
    
    def getData(self):
        if self.isMarketOpen():
            print("Stock Data is Downloading")
            data = yf.download(self.ticker, period=self.period, interval=self.interval)
        else:
            if 'data' in self.data_cache:
                print("Stock Data found in cache!")
                data = self.data_cache['data']
            else:
                print("Stock Data is Downloading")
                data = yf.download(self.ticker, period=self.period, interval=self.interval)
                self.data_cache['data'] = data

        return data


class DataProcessing:
    def __init__(self, ticker, data):
        self.ticker = ticker
        self.data = data
        self.data = self.data[['Open']]
        self.data = self.data.dropna(axis=0, how='any')
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.last = None
        self.window = 5
        self.n_future = 12
        self.sc = MinMaxScaler(feature_range = (0, 1))
        self.ratio = [0.75, 0.15, 0.1]
        

    def preprocessing(self):
        self.training_set_scaled = self.sc.fit_transform(self.data[['Open']])
        x_train, y_train, last = [], [], []


        for window_start in range(len(self.training_set_scaled)):
            past_end = window_start + self.window
            future_end = past_end + self.n_future
            if future_end > len(self.training_set_scaled):
                last_id = window_start
                break
            # slicing the past and future parts of the window
            past, future = self.training_set_scaled[window_start:past_end, :], self.training_set_scaled[past_end:future_end, 0]
            x_train.append(past)
            y_train.append(future)

        for window_start in range(last_id, len(self.training_set_scaled)):
            past_end = window_start + self.window
            if past_end > len(self.training_set_scaled):
                break
            last.append(self.training_set_scaled[window_start:past_end, :])

        if(self.window + len(x_train) + len(last) == self.training_set_scaled.shape[0]+1):
            print("Data Reading Done!")
        else:
            print("Data reading is not done properly!")
        
        x_train, y_train = np.array(x_train), np.array(y_train)


        trainNo = int(x_train.shape[0]*self.ratio[0])
        valNo = int(x_train.shape[0]*self.ratio[1])

        self.test_data = (x_train[trainNo + valNo:], y_train[trainNo + valNo:])
        self.val_data = (x_train[trainNo: trainNo + valNo], y_train[trainNo: trainNo + valNo])
        self.train_data = (x_train[:trainNo], y_train[:trainNo])


class ModelBuilding:
    def __init__(self, ticker, path, window_size=5):
        self.ticker = ticker
        self.dir_path = path
        self.model_path = self.dir_path + '/LiveModel.h5'
        self.input_shape = (window_size, 1)
        self.nepoch = 30
        self.batch_size = 8
        self.cache = {}

    def createModel(self, n_future):
        modelObject = ShortTermPrediction(n_future=n_future)
        self.model = modelObject.buildModel(input_shape=self.input_shape)
        
    def training(self, train_data, val_data, epochs=None):
        if epochs is not None:
            self.nepoch = epochs

        self.history = self.model.fit(train_data[0], train_data[1], validation_data=val_data, epochs=self.nepoch, batch_size=self.batch_size, callbacks=[ModelCheckpoint(self.model_path, monitor='val_mape', save_best_only=True)])
        self.train_loss = self.model.evaluate(train_data[0], train_data[1])
        self.val_loss = self.model.evaluate(val_data[0], val_data[1])
        self.log_path = self.dir_path + '/LiveStockModelEvaluation.txt'
        with open(self.log_path, 'w') as file:
            file.write("Live Stock Prediction Model Evaluation.\nMetrices are: MSE, MAPE, MAE")
            file.write("Train Metrices: {}\nValidation Metrices: {}\n".format(self.train_loss, self.val_loss))
        file.close()
        self.cache['model'] = self.model


    def fineTuning(self, train_data, val_data):
        optimizer = Adam(1e-5)
        self.model = self.cache['model']
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mape', 'mae'])
        self.history = self.model.fit(train_data[0], train_data[1], validation_data=val_data, epochs=5, batch_size=self.batch_size, callbacks=[ModelCheckpoint(self.model_path, monitor='val_mape', save_best_only=True)])
        self.cache['model'] = self.model
    
    def testing(self, test_data):
        self.test_loss = self.model.evaluate(test_data[0], test_data[1])
        with open(self.log_path, 'a') as file:
            file.write("Test Metrices: {}\n".format(self.test_loss))
        file.close()

    




if __name__=="__main__":
    ADANIPOWER = DataDownloader('ADANIPOWER.NS')
    start_time = ADANIPOWER.currentTime()
    time_change = timedelta(minutes=5)
    firstTimeFlag = True

    ADANIPOWER_Processing = DataProcessing(ADANIPOWER.ticker, ADANIPOWER.getData())
    ADANIPOWER_Processing.preprocessing()
    ADANIPOWER_Model = ModelBuilding(ADANIPOWER.ticker, './')
    ADANIPOWER_Model.createModel(ADANIPOWER_Processing.n_future)
    
    while True:
        if firstTimeFlag:
            ADANIPOWER_Model.training(ADANIPOWER_Processing.train_data, ADANIPOWER_Processing.val_data)
            firstTimeFlag = False
        else:
            ADANIPOWER_Processing = DataProcessing(ADANIPOWER.ticker, ADANIPOWER.getData())
            ADANIPOWER_Processing.preprocessing()
            ADANIPOWER_Model.fineTuning(ADANIPOWER_Processing.train_data, ADANIPOWER_Processing.val_data)
        
        print("----------------------------------------------------------Let Me Sleep for 2 minutes-------------------------------------------------------------------------")
        time.sleep(120)




