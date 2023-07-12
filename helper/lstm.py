import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

class LSTMModel:
  def __init__(self, path):
    self.model = None
    self.model_summary = None
    self.history = None
    self.batch_size = 16
    self.sc = None
    self.train_loss = None
    self.val_loss = None
    self.test_loss = None
    self.train_mse = None
    self.val_mse = None
    self.test_mse = None
    self.model_path = path
    self.n_future = 1
    self.n_features = 2

  def buildModel(self, input_shape = (10, 1)):
    encoder_inputs = Input(shape=input_shape)
    encoder_l1 = LSTM(16, return_state=True)
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]

    decoder_inputs = RepeatVector(self.n_future)(encoder_outputs1[0])
    decoder_l1 = LSTM(16, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
    decoder_outputs1 = TimeDistributed(Dense(self.n_features))(decoder_l1)
    decoder_outputs1 = Dense(self.n_features)(decoder_l1)

    model = Model(encoder_inputs,decoder_outputs1)
    model.compile(optimizer=Adam(), loss='mean_absolute_error')
    self.model = model
    self.model_summary = model.summary()
    print(self.model_summary)
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

  def fitModel(self, train_data, val_data=None, epochs=20, batch_size=16):
    self.history = self.model.fit(train_data[0], train_data[1], validation_data=val_data, epochs=epochs, batch_size=self.batch_size, callbacks=[ModelCheckpoint(self.model_path, monitor='val_mse', save_best_only=True)])
  
  def evaluateModel(self, data):
    return self.model.evaluate(data[0], data[1])

  def predictModel(self, xdata):
     predicted = self.model.predict(xdata)
     if(len(predicted.shape)>2):
       return self.sc.inverse_transform(np.reshape(predicted, newshape=(predicted.shape[0], -1)))
     else:
       return self.sc.inverse_transform(predicted)

  def evaluateMSE(self, data):
    pred = self.model.predict(data[0])
    if(pred.shape[1]==1):
      pred = np.reshape(pred, newshape=(pred.shape[0], 2))
    predicted = self.sc.inverse_transform(pred)
    if(data[1].shape[1]==1):
      return mean_squared_error(np.reshape(data[1], newshape=(data[1].shape[0], 2)), predicted)
    else:
      return mean_squared_error(data[1], predicted)
  
  def LoadModel(self, filepath):
    return load_model(filepath)

