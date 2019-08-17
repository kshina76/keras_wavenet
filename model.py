import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, Multiply, Add, Lambda, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.regularizers import l2

from keras.layers.convolutional import Conv2D


class model(object):
	def __init__(self, receptive_field, r_channels,
				s_channels, a_channel, d_channels, n_layer, n_loop):
		self.filter_size = 2
		self.img_rows = receptive_field
		self.img_columns = 1
		self.a_channel = a_channel
		self.r_channels = r_channels
		self.s_channels = s_channels
		self.d_channels = d_channels
		self.n_loop = n_loop
		self.n_layer = n_layer
		self.dilation = [2 ** i for i in range(n_layer)] * n_loop

	def ResidualBlock(self, block_in, dilation_index):
		res = block_in
		tanh_out = Conv2D(self.d_channels, (self.filter_size, 1), padding='same',
						dilation_rate=(dilation_index, 1), activation='tanh')(block_in)
		sigm_out = Conv2D(self.d_channels, (self.filter_size, 1), padding='same',
						dilation_rate=(dilation_index, 1), activation='sigmoid')(block_in)
		marged = Multiply()([tanh_out, sigm_out])
		res_out = Conv2D(self.r_channels, (1,1), padding='same')(marged)
		skip_out = Conv2D(self.s_channels, (1,1), padding='same')(marged)
		res_out = Add()([res_out,res])

		return res_out, skip_out

	def ResidualNet(self, block_in):
		skip_out_list = []
		for dilation_index in self.dilation:
			res_out, skip_out = self.ResidualBlock(block_in, dilation_index)
			skip_out_list.append(skip_out)
			block_in = res_out

		return skip_out_list

	def wavenet(self):
		inputs = Input(shape=(self.img_rows, self.img_columns, self.a_channel))
		causal_conv = Conv2D(self.r_channels, (self.filter_size, 1), padding='same')(inputs)
		skip_out_list = self.ResidualNet(causal_conv)
		skip_out = Add()(skip_out_list)
		skip_out = Activation('relu')(skip_out)
		skip_out = Conv2D(self.a_channel, (1,1), padding='same', activation='relu')(skip_out)
		prediction = Conv2D(self.a_channel, (1,1), padding='same')(skip_out)
		prediction = Flatten()(prediction)
		prediction = Dense(self.a_channel, activation='softmax')(prediction)

		model_wavenet = Model(input=inputs,output=prediction)
		optimizer = Adam()
		model_wavenet.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
		model_wavenet.summary()

		return model_wavenet