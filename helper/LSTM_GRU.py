import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import Variable
import math
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, RepeatVector, TimeDistributed, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import plot_model

class ShortTermPrediction:
    def __init__(self, n_future):
        self.model = None
        self.model_summary = None
        self.n_future = n_future
        self.target_num = 1

    
    def buildModel(self, input_shape = (5, 1)):
        encoder_inputs = Input(shape=input_shape, name="Input_Layer")
        LSTM_encoder_l1 = LSTM(4, return_state=True, name="Encoder_LSTM_Layer")
        GRU_encode_l1 = GRU(4, return_state=True, name="Encode_GRU_layer")
        LSTM_encoder_outputs1 = LSTM_encoder_l1(encoder_inputs)
        GRU_encoder_outputs1 = GRU_encode_l1(encoder_inputs)
        LSTM_encoder_states1 = LSTM_encoder_outputs1[1:]
        #GRU_encoder_states1 = GRU_encoder_outputs1[1:]

        LSTM_decoder_inputs = RepeatVector(self.n_future, name="Reapet_Layer_LSTM")(LSTM_encoder_outputs1[0])
        GRU_decoder_inputs = RepeatVector(self.n_future, name="Reapet_Layer_GRU")(GRU_encoder_outputs1[0])
        LSTM_decoder_l1 = LSTM(4, return_sequences=True, name="Decoder_LSTM_Layer")(LSTM_decoder_inputs, initial_state = LSTM_encoder_states1)
        GRU_decoder_l1 = GRU(4, return_sequences=True, name="Decoder_GRU_Layer")(GRU_decoder_inputs, initial_state=GRU_encoder_outputs1[-1])
        Concatenate_lyer = Concatenate(axis=-1)([LSTM_decoder_l1, GRU_decoder_l1])
        decoder_outputs1 = TimeDistributed(Dense(self.target_num, name="Dense_Layer"), name="Time_Distributed_Layer")(Concatenate_lyer)
        decoder_outputs1 = Dense(self.target_num, name="Output_Layer")(decoder_outputs1)

        model = Model(encoder_inputs,decoder_outputs1)
 

        
        optimizer = Adam(1e-4)
        model.compile(optimizer, loss='mse', metrics=['mape', 'mae'])
        self.model = model
        self.model_summary = model.summary()
        plot_model(model, to_file='short_model_plot.png', show_shapes=True, show_layer_names=True)

        return self.model

    