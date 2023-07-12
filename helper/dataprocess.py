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
  def __init__(self, file_path, columnns=[], change=False):
    self.data_path = file_path
    self.original_data = pd.read_csv(self.data_path)
    #self.data[columnns] = self.data[columnns].rolling(5).mean()
    self.data = self.original_data[columnns]
    self.data = self.data.dropna(axis=0, how='any')
    #self.data = self.data.diff()[columnns]
    #self.data = self.data.dropna(axis=0, how='any')
    print(self.data)
    self.data = self.data.values
    self.x_train = []
    self.y_train = []
    self.x_test = []
    self.y_test = []
    self.x_val = []
    self.y_val = []
    self.last = None


  def split(self, window=5, n_future=1, ratio=[0.7, 0.1, 0.1]):
    self.sc = MinMaxScaler(feature_range = (0, 1))
    self.training_set_scaled = self.sc.fit_transform(self.data)
    #self.training_set_scaled = self.data
    print(self.training_set_scaled.shape)
    
    for i in range(window, len(self.training_set_scaled)):
      self.x_train.append(self.training_set_scaled[i-window: i, :])
      self.y_train.append(self.training_set_scaled[i:i+1, :])

      """
    for window_start in range(len(self.training_set_scaled)):
      past_end = window_start + window
      future_end = past_end + n_future
      if future_end > len(self.training_set_scaled):
        break
      # slicing the past and future parts of the window
      past, future = self.training_set_scaled[window_start:past_end, :], self.training_set_scaled[past_end:future_end, :]
      self.x_train.append(past)
      self.y_train.append(future)
"""
    print(self.x_train[0])
    print(self.x_train[1])
    print(self.y_train[0])
    print(self.y_train[1])
      

    self.last = self.training_set_scaled[len(self.training_set_scaled)-1, :]
    self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
    #self.x_train = np.reshape(self.x_train, newshape = (self.x_train.shape[0], self.x_train.shape[1], 1))

    if len(ratio)==3:
      print("Shape of x is ", self.x_train.shape, " Shape of y is ", self.y_train.shape)
      trainNo = int(self.x_train.shape[0]*ratio[0])
      valNo = trainNo + int(self.x_train.shape[0]*ratio[1])

      self.x_test = self.x_train[valNo:]
      self.y_test = self.y_train[valNo:]
      self.x_val = self.x_train[trainNo: valNo]
      self.y_val = self.y_val[trainNo: valNo]
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
