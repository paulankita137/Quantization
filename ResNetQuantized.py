# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 15:58:16 2021

@author: ANKITA PAUL
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from collections import defaultdict
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

from qkeras import QActivation
from qkeras import QDense
from qkeras import quantizers
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings
import argparse
from qkeras import *
from qkeras.utils import model_save_quantized_weights
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

np.random.seed(42)

NB_EPOCH = 10
BATCH_SIZE = 500
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam(lr=0.0001, decay=0.000025)
VALIDATION_SPLIT = 0.1

def relu_bn(inputs: Tensor) -> Tensor:
    #relu = ReLU()(inputs)
    relu=QActivation("quantized_relu(2,0)")(inputs)

    bn = QBatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = QConv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same",kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),)(x)
    y = relu_bn(y)
    y = QConv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same",kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),)(y)

    if downsample:
        x = QConv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same",kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),)(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net():
    
    inputs = Input(shape=(32, 32, 3))
    x = x_in = Input(shape=(32, 32, 3), name="input")

    num_filters = 64
    
    t = QBatchNormalization()(x)
    t = QConv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same",kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),)(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 5, 5, 2]
    num_blocks_list = [2]

    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    
    t = QAveragePooling2D(4)(t)
    t = Flatten()(t)
    x = QDense(10,kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),)(t)
    x_out=x
    x= Activation("softmax", name="softmax")(x)

    
    # model = Model(inputs, outputs)

    # model.compile(
    #     optimizer='adam',
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy']
    # )
    
    
    
    
    model = Model(inputs=[x_in], outputs=[x])
    mo = Model(inputs=[x_in], outputs=[x_out])
    model.summary()

    model.compile(loss="sparse_categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

    model.save('ResnetCIFARQuantized2bit.h5')
    
    return model







#Running Resnet

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import os

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = create_res_net() # or create_plain_net()
model.summary()

timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
name = 'cifar-10_res_net_30-'+timestr # or 'cifar-10_plain_net_30-'+timestr

checkpoint_path = "checkpoints/"+name+"/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
os.system('mkdir {}'.format(checkpoint_dir))

# save model after each epoch
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1
)
tensorboard_callback = TensorBoard(
    log_dir='tensorboard_logs/'+name,
    histogram_freq=1
)

model.fit(
    x=x_train,
    y=y_train,
    epochs=20,
    verbose=1,
    validation_data=(x_test, y_test),
    batch_size=500,
    callbacks=[cp_callback, tensorboard_callback]
)

score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])
all_weights = []
model_save_quantized_weights(model)

for layer in model.layers:
    for w, weights in enumerate(layer.get_weights()):
      print(layer.name, w)
      all_weights.append(weights.flatten())

all_weights = np.concatenate(all_weights).astype(np.float32)
print(all_weights.size)
all_weights = []
model_save_quantized_weights(model)

for layer in model.layers:
    for w, weights in enumerate(layer.get_weights()):
      print(layer.name, w)
      all_weights.append(weights.flatten())

all_weights = np.concatenate(all_weights).astype(np.float32)
print(all_weights.size)


for layer in model.layers:
  for w, weight in enumerate(layer.get_weights()):
    print(layer.name, w, weight.shape)

print_qstats(model)
quantizedweights=model_save_quantized_weights(model)
model_save_quantized_weights(model)
import csv

my_dict = quantizedweights
np.save('quantizedweightsResNetcifar10.npy', my_dict) 


model.save('ResNetCIFARQuantized2bit.h5')

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



def hybrid_model():
  """hybrid model that mixes qkeras and keras layers."""

  
  x = x_in = Input(x_train.shape[1:], name="input")
  # x = QActivation("quantized_relu_po2(4,4)", name="acti")(x)
  #Layer1
  x = QConv2D(
    96, (11, 11),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_0_m")(x)
  x = BatchNormalization()(x)
  x = QActivation("quantized_relu(2,0)", name="act0_m")(x)
  x= MaxPool2D(pool_size=(2,2),data_format='channels_last')(x)

  #Layer2
  x = QConv2D(
    256, (5, 5),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_1_m")(x)
  x = BatchNormalization()(x)
  x = QActivation("quantized_relu(2,0)", name="act1_m")(x)
  x= MaxPool2D(pool_size=(2,2),data_format='channels_last')(x)

  #Layer3

  x = QConv2D(
    512, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_2_m")(x)
  x = BatchNormalization()(x)
  x = QActivation("quantized_relu(2,0)", name="act2_m")(x)
  x= MaxPool2D(pool_size=(2,2),data_format='channels_last')(x)


 #Layer4

  x = QConv2D(
    1024, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_3_m")(x)
  x = BatchNormalization()(x)
  x = QActivation("quantized_relu(2,0)", name="act3_m")(x)
  x= MaxPool2D(pool_size=(2,2),data_format='channels_last')(x)


  #Layer5
  x = QConv2D(
    1024, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_4_m")(x)
  x = BatchNormalization()(x)
  x = QActivation("quantized_relu(2,0)", name="act4_m")(x)
  x= MaxPool2D(pool_size=(1,1),data_format='channels_last')(x)

  #Layer6
  x = Flatten()(x)
  x = QDense(3072, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense0")(x)
  x = BatchNormalization()(x)
  x = QActivation("quantized_relu(2,0)", name="act5_m")(x)
  x = Dropout(0.5)(x)
  #Layer7

  x = QDense(4096, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense1")(x)
  x = BatchNormalization()(x)

  x = QActivation("quantized_relu(2,0)", name="act6_m")(x)

  x = Dropout(0.5)(x)

  #Layer8

  x = QDense(10, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense2")(x)
  x = BatchNormalization()(x)

  x_out = x
  x = Activation("softmax", name="softmax")(x)

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
