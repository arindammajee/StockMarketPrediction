import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import Variable
import math
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import plot_model

class LSTMModel:
  def __init__(self, n_future, target_num, path):
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
    self.n_future = n_future
    self.target_num = target_num
    self.epochs = 20

  
  
  def buildModel(self, input_shape = (10, 1)):
    encoder_inputs = Input(shape=input_shape, name="Input_Layer")
    encoder_l1 = LSTM(16, return_state=True, name="Encoder_LSTM_Layer")
    encoder_outputs1 = encoder_l1(encoder_inputs)
    encoder_states1 = encoder_outputs1[1:]

    decoder_inputs = RepeatVector(self.n_future, name="Reapet_Layer")(encoder_outputs1[0])
    decoder_l1 = LSTM(16, return_sequences=True, name="Decoder_LSTM_Layer")(decoder_inputs,initial_state = encoder_states1)
    decoder_outputs1 = TimeDistributed(Dense(self.target_num, name="Dense_Layer"), name="Time_Distributed_Layer")(decoder_l1)
    decoder_outputs1 = Dense(self.target_num, name="Output_Layer")(decoder_l1)

    model = Model(encoder_inputs,decoder_outputs1)

    # Later, whenever we perform an optimization step, we pass in the step.
    def get_lr_metric(optimizer):
      def lr(y_true, y_pred):
          return optimizer.lr
      return lr
    
    optimizer = Adam(1e-4)
    lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer, loss='mse', metrics=['mape', 'mae'])
    self.model = model
    self.model_summary = model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

  def fitModel(self, train_data, val_data=None, epochs=250):    
    self.history = self.model.fit(train_data[0], train_data[1], validation_data=val_data, epochs=epochs, batch_size=self.batch_size, callbacks=[ModelCheckpoint(self.model_path, monitor='val_mape', save_best_only=True)])
  
  def evaluateModel(self, data):
    return self.model.evaluate(data[0], data[1])

  def LoadModel(self, filepath):
      return load_model(filepath)
  
  def fineTuneModel(self, train_data, val_data=None, epochs=100):
    optimizer = Adam(1e-5)
    self.model.compile(optimizer=optimizer, loss='mse', metrics=['mape', 'mae'])
    self.history = self.model.fit(train_data[0], train_data[1], validation_data=val_data, epochs=epochs, batch_size=self.batch_size, callbacks=[ModelCheckpoint(self.model_path, monitor='val_mape', save_best_only=True)])
  
