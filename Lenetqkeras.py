# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 19:10:55 2021

@author: ANKITA PAUL
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:58:16 2021

@author: ANKITA PAUL
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras as keras

from qkeras import QActivation
from qkeras import QDense
from qkeras import quantizers
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings
import argparse

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np

from qkeras import print_qstats
from qkeras import QActivation
from qkeras import QDense
from qkeras import quantized_bits
from qkeras import ternary
import io
import setuptools
#golden commands to run the setuptools
import os
from collections import defaultdict

import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

from qkeras import *
from qkeras.utils import model_save_quantized_weights
import numpy as np
import tensorflow.compat.v1 as tf

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import std
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,BatchNormalization,Flatten,Activation, Dropout, Input
from tensorflow import keras
import numpy as np
import pandas as pd
import gc
import random
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, Nadam,SGD
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#Dataset
df_train = pd.read_csv('mnist_train.csv')

# dataset=pd.read_csv('FreshDVT2.csv')
# dataset2=pd.read_csv('DVTTarget1.csv')
X_train=df_train.iloc[:, 1:]


df_test = pd.read_csv('D:\DATE\mnist_test.csv')
X_test = df_test.iloc[:, 1:]

X_test = np.array(X_test)
y_test=df_test.iloc[:, 0:]
#X=X.reshape(1,101113,2)
print(X.shape)

y_train=df_train.iloc[:, 0:]

X_train = X_train / 255.0


# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=100)
# X_testt=np.asarray(X_test)
# y_testt=np.asarray(y_test).T
# y_testt=y_testt.T.reshape(-1,1)
# print(y_testt)
# X_traint=np.asarray(X_train)



np.random.seed(50)
OPTIMIZER = Adam()
NB_EPOCH = 5
BATCH_SIZE = 25
VERBOSE = 1
NB_CLASSES = 1
N_HIDDEN1 = 200
N_HIDDEN2 = 100

VALIDATION_SPLIT = 0.1
RESHAPED = 2


from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=50)
x_train=X_train
y_train=y_train
y_test=y_test
x_test=x_test
RESHAPED = 1

x_test_orig = x_test

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

X_train = x_train[..., np.newaxis]
X_test = x_test[..., np.newaxis]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
# x_train /= 256.0
# x_test /= 256.0

print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


def QDenseModel(weights_f, load_weights=False):
  """Construct QDenseModel."""

  # x = x_in = Input((RESHAPED,), name="input")
  # #x = QActivation("quantized_relu(4)", name="act_i")(x)
  # x = QDense(N_HIDDEN1, kernel_quantizer=ternary(),
  #            bias_quantizer=quantized_bits(32, 0, 1), name="dense0")(x)
  # x = QActivation("quantized_relu(2)", name="act0")(x)
  # x = QDense(
  #     N_HIDDEN2,
  #     kernel_quantizer=quantized_bits(32, 0, 1),
  #     bias_quantizer=quantized_bits(32, 0, 1),
  #     name="dense2")(
  #         x)
  # x = QDense(
  #     NB_CLASSES,
  #     kernel_quantizer=quantized_bits(32, 0, 1),
  #     bias_quantizer=quantized_bits(32, 0, 1),
  #     name="dense3")(
  #         x)
  # x = Activation("sigmoid", name="sigmoid")(x)

  # model = Model(inputs=[x_in], outputs=[x])
  # model.summary()
  # model.compile(loss="binary_crossentropy",
  #               optimizer=OPTIMIZER, metrics=["accuracy"])
  
  
  
  # x = x_in = Input(
  #    X_train.shape[1:-1] + (1,), name="input")
  # print('x babe',x)
  x = x_in = Input(
     X_train.shape[1:] , name="input")
  #x = x_in = Input((3,), name="input")
  x = QConv1D(
    64, (1), strides=(1),
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv1d_0_m")(x)
  x = QActivation("quantized_relu(4,0)", name="act0_m")(x)
  x= MaxPool1D(pool_size=1)(x)
  x = Flatten()(x)
  # x = MaxPooling1D(1)(x)

  x = QDense(N_HIDDEN1, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense1")(x)
  x = QActivation("quantized_relu(4,0)", name="act1_m")(x)
  x = QDense(N_HIDDEN2, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense2")(x)

  x = QActivation("sigmoid", name="sigmoid1")(x)

  x = QDense(1, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense3")(x)
  x = Activation("sigmoid", name="sigmoid")(x)
  model = Model(inputs=[x_in], outputs=[x])
  model.summary()

  model.compile(
    loss="binary_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
  
  
  
  
  
  #print_qstats(model)
  if load_weights and weights_f:
     model.load_weights(weights_f)

     print_qstats(model)
  return model


def UseNetwork(weights_f, load_weights=False):
  """Use DenseModel.

  Args:
    weights_f: weight file location.
    load_weights: load weights when it is True.
  """
  model = QDenseModel(weights_f, load_weights)

  batch_size = BATCH_SIZE
  dataset=pd.read_csv('FreshDVT2.csv')
  dataset2=pd.read_csv('DVTTarget1.csv')
  X=dataset.iloc[:,0:2].values
  #X=X.reshape(1,101113,2)
  print(X.shape)

  y=dataset2.iloc[:,0].values



  from sklearn.model_selection import train_test_split
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=50)

  print(X_train.shape[0], "train samples")
  print(X_test.shape[0], "test samples")

  

  if not load_weights:
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=NB_EPOCH,
        verbose=VERBOSE,
        validation_split=VALIDATION_SPLIT)

    if weights_f:
      model.save_weights(weights_f)

  score = model.evaluate(X_test, y_test, verbose=VERBOSE)
  #print_qstats(model)
  print("Test score:", score[0])
  print("Test accuracy:", score[1])


def ParserArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument("-l", "--load_weight", default="0",
                      help="""load weights directly from file.
                            0 is to disable and train the network.""")
  parser.add_argument("-w", "--weight_file", default=None)
  a = parser.parse_args()
  return a


if __name__ == "__main__":
  args = ParserArgs()
  lw = False if args.load_weight == "0" else True
  UseNetwork(args.weight_file, load_weights=lw)
  
  
  
#Energy generation with QTools


def hybrid_model():
  """hybrid model that mixes qkeras and keras layers."""

  
  x = x_in = Input(
     X_train.shape[1:] , name="input")
  #x = x_in = Input((3,), name="input")
  x = QConv1D(
    64, (1), strides=(1),
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv1d_0_m")(x)
  x = QActivation("quantized_relu(4,0)", name="act0_m")(x)
  x= MaxPool1D(pool_size=1)(x)
  x = Flatten()(x)
  # x = MaxPooling1D(1)(x)

  x = QDense(N_HIDDEN1, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense1")(x)
  x = QActivation("quantized_relu(4,0)", name="act1_m")(x)
  x = QDense(N_HIDDEN2, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense2")(x)

  #x = QActivation("sigmoid", name="sigmoid1")(x)

  x = QDense(1, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense3")(x)
  x = Activation("sigmoid", name="sigmoid")(x)
  return keras.Model(inputs=[x_in], outputs=[x])

if __name__ == "__main__":
  # input parameters:
  # process: technology process to use in configuration (horowitz, ...)
  # weights_on_memory: whether to store parameters in dram, sram, or fixed
  # activations_on_memory: store activations in dram or sram
  # rd_wr_on_io: whether load data from dram to sram (consider sram as a cache
  #   for dram. If false, we will assume data will be already in SRAM
  # source_quantizers: quantizers for model input
  # is_inference: whether model has been trained already, which is
  #   needed to compute tighter bounds for QBatchNormalization Power estimation.
  # reference_internal: size to use for weight/bias/activation in
  #   get_reference energy calculation (int8, fp16, fp32)
  # reference_accumulator: accumulator and multiplier type in get_reference
  #   energy calculation
  model = hybrid_model()
  model.summary()

  reference_internal = "int8"
  reference_accumulator = "int32"

  # By setting for_reference=True, we create QTools object which uses
  # keras_quantizer to quantize weights/bias and
  # keras_accumulator to quantize MAC variables for all layers. Obviously, this
  # overwrites any quantizers that user specified in the qkeras layers. The
  # purpose of doing so is to enable user to calculate a baseline energy number
  # for a given model architecture and compare it against quantized models.
  q = run_qtools.QTools(
      model,
      # energy calculation using a given process
      process="horowitz",
      # quantizers for model input
      source_quantizers=[quantizers.quantized_bits(2, 0, 1)],
      is_inference=False,
      # absolute path (including filename) of the model weights
      weights_path=None,
      # keras_quantizer to quantize weight/bias in un-quantized keras layers
      keras_quantizer=reference_internal,
      # keras_quantizer to quantize MAC in un-quantized keras layers
      keras_accumulator=reference_accumulator,
      # whether calculate baseline energy
      for_reference=True)

  # caculate energy of the derived data type map.
  ref_energy_dict = q.pe(
      # whether to store parameters in dram, sram, or fixed
      weights_on_memory="sram",
      # store activations in dram or sram
      activations_on_memory="sram",
      # minimum sram size in number of bits
      min_sram_size=8*16*1024*1024,
      # whether load data from dram to sram (consider sram as a cache
      # for dram. If false, we will assume data will be already in SRAM
      rd_wr_on_io=False)

  # get stats of energy distribution in each layer
  reference_energy_profile = q.extract_energy_profile(
      qtools_settings.cfg.include_energy, ref_energy_dict)
  # extract sum of energy of each layer according to the rule specified in
  # qtools_settings.cfg.include_energy
  total_reference_energy = q.extract_energy_sum(
      qtools_settings.cfg.include_energy, ref_energy_dict)
  print("Baseline energy profile:", reference_energy_profile)
  print("Total baseline energy:", total_reference_energy)

  # By setting for_reference=False, we quantize the model using quantizers
  # specified by users in qkeras layers. For hybrid models where there are
  # mixture of unquantized keras layers and quantized qkeras layers, we use
  # keras_quantizer to quantize weights/bias and keras_accumulator to quantize
  # MAC variables for all keras layers.
  q = run_qtools.QTools(
      model, process="horowitz",
      source_quantizers=[quantizers.quantized_bits(2, 0, 1)],
      is_inference=False, weights_path=None,
      keras_quantizer=reference_internal,
      keras_accumulator=reference_accumulator,
      for_reference=False)
  trial_energy_dict = q.pe(
      weights_on_memory="sram",
      activations_on_memory="sram",
      #min_sram_size=8*16*1024*1024,
      min_sram_size=2*16*1024*1024,

      rd_wr_on_io=False)
  trial_energy_profile = q.extract_energy_profile(
      qtools_settings.cfg.include_energy, trial_energy_dict)
  total_trial_energy = q.extract_energy_sum(
      qtools_settings.cfg.include_energy, trial_energy_dict)
  print("energy profile:", trial_energy_profile)
  print("Total energy:", total_trial_energy)
