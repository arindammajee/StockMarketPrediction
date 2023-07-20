import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


class DataProcess:
  def __init__(self, file_path, window, n_future, columnns=[]):
    self.data_path = file_path
    self.original_data = pd.read_csv(self.data_path)
    self.data = self.original_data[self.original_data.columns[:5]]
    #self.data[columnns] = self.data[columnns].rolling(5).mean()
    self.data = self.data.dropna()
    self.last_date = self.data['Date'].iloc[-1][:10]
    self.data = self.data[columnns]
    #self.data = self.data.diff()[columnns]
    #self.data = self.data.dropna(axis=0, how='any')
    self.data = self.data.values
    self.x_train = []
    self.y_train = []
    self.x_test = []
    self.y_test = []
    self.x_val = []
    self.y_val = []
    self.last = None
    self.n_future = n_future
    self.window = window
    self.n_feature = len(columnns)
    self.last = []

  def split(self, ratio=[0.7, 0.1, 0.1]):
    self.sc = MinMaxScaler(feature_range = (0, 1))
    self.training_set_scaled = self.sc.fit_transform(self.data)
    print(self.training_set_scaled.shape)
    """
    for i in range(self.window, len(self.training_set_scaled)-self.n_future + 1):
      self.x_train.append(self.training_set_scaled[i-self.window: i, :])
      self.y_train.append(self.training_set_scaled[i: i+self.n_future, 0])

    for i in range(len(self.training_set_scaled)-self.n_future, len(self.training_set_scaled)):
      self.last.append(self.training_set_scaled[i, :])
      """
    for window_start in range(len(self.training_set_scaled)):
      past_end = window_start + self.window
      future_end = past_end + self.n_future
      if future_end > len(self.training_set_scaled):
        last_id = window_start
        break
      # slicing the past and future parts of the window
      past, future = self.training_set_scaled[window_start:past_end, :], self.training_set_scaled[past_end:future_end, 0]
      self.x_train.append(past)
      self.y_train.append(future)

    for window_start in range(last_id, len(self.training_set_scaled)):
      past_end = window_start + self.window
      if past_end > len(self.training_set_scaled):
        break
      self.last.append(self.training_set_scaled[window_start:past_end, :])
    print(len(self.last))
    if(self.window + len(self.x_train) + len(self.last) == self.training_set_scaled.shape[0]+1):
      print("Data Reading Done!")
    else:
      print("Data reading is not done properly!")
    
    self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
    

    if len(ratio)==3:
      print("Shape of x is ", self.x_train.shape, " Shape of y is ", self.y_train.shape)
      trainNo = int(self.x_train.shape[0]*ratio[0])
      valNo = int(self.x_train.shape[0]*ratio[1])

      self.x_test = self.x_train[trainNo + valNo:]
      self.y_test = self.y_train[trainNo + valNo:]
      self.x_val = self.x_train[trainNo: trainNo + valNo]
      self.y_val = self.y_train[trainNo: trainNo + valNo]
      self.x_train = self.x_train[:trainNo]
      self.y_train = self.y_train[:trainNo]
      
      print("Shape of x_train is ", self.x_train.shape, " Shape of y-train is ", self.y_train.shape)
      print("Shape of x_val is ", self.x_val.shape, " Shape of y_val is ", self.y_val.shape)
      print("Shape of x_test is ", self.x_test.shape, " Shape of y_test is ", self.y_test.shape)
    
    elif len(ratio)==2:
      print("Shape of x is ", self.x_train.shape, " Shape of y is ", self.y_train.shape)
      trainNo = int(self.x_train.shape[0]*ratio[0])

      self.x_test = self.x_train[trainNo:]
      self.y_test = self.y_train[trainNo:]
      self.x_train = self.x_train[:trainNo]
      self.y_train = self.y_train[:trainNo]
      
      print("Shape of x_train is ", self.x_train.shape, " Shape of y-train is ", self.y_train.shape)
      print("Shape of x_test is ", self.x_test.shape, " Shape of y_test is ", self.y_test.shape)
