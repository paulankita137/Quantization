# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 18:15:44 2021

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

np.random.seed(42)

NB_EPOCH = 10
BATCH_SIZE = 2000
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam(lr=0.0001, decay=0.000025)
VALIDATION_SPLIT = 0.1

train = 1

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

RESHAPED = 784

x_test_orig = x_test

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")


#Enable the following two lines for mnist
# x_train = x_train[..., np.newaxis]
# x_test = x_test[..., np.newaxis]

x_train /= 255.0
x_test /= 255.0

print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

print(y_train[0:10])

y_train = to_categorical(y_train, NB_CLASSES)
y_test = to_categorical(y_test, NB_CLASSES)

# x = x_in = Input(
#     x_train.shape[1:-1] + (1,), name="input")

x = x_in = Input(x_train.shape[1:], name="input")
# x = QActivation("quantized_relu_po2(4,4)", name="acti")(x)
#Layer1
x = QConv2D(
    64, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_0_m")(x)
x = QActivation("quantized_relu(2,0)", name="act0_m")(x)

#Layer2
x = QConv2D(
    64, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_1_m")(x)
x = QActivation("quantized_relu(2,0)", name="act1_m")(x)
x= MaxPool2D(pool_size=(2,2),strides=(2,2),data_format='channels_last')(x)

#Layer3

x = QConv2D(
    128, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_2_m")(x)
x = QActivation("quantized_relu(2,0)", name="act2_m")(x)

#Layer4

x = QConv2D(
    128, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_3_m")(x)
x = QActivation("quantized_relu(2,0)", name="act3_m")(x)
x= MaxPool2D(pool_size=(2,2),strides=(2,2),data_format='channels_last')(x)


#Layer5
x = QConv2D(
    256, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_4_m")(x)
x = QActivation("quantized_relu(2,0)", name="act4_m")(x)

x = QConv2D(
    256, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_5_m")(x)
x = QActivation("quantized_relu(2,0)", name="act5_m")(x)

x = QConv2D(
    256, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_6_m")(x)
x = QActivation("quantized_relu(2,0)", name="act6_m")(x)
x= MaxPool2D(pool_size=(2,2),strides=(2,2),data_format='channels_last')(x)

x = QConv2D(
    512, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_7_m")(x)
x = QActivation("quantized_relu(2,0)", name="act7_m")(x)

x = QConv2D(
    512, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_8_m")(x)
x = QActivation("quantized_relu(2,0)", name="act8_m")(x)

x = QConv2D(
    512, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_9_m")(x)
x = QActivation("quantized_relu(2,0)", name="act9_m")(x)
x=  MaxPool2D(pool_size=(2,2),strides=(2,2),data_format='channels_last')(x)



x = Flatten()(x)
x = QDense(4096, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense1")(x)
x = QActivation("quantized_relu(2,0)", name="act10_m")(x)

x = QDense(4096, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense2")(x)
x = QActivation("quantized_relu(2,0)", name="act11_m")(x)

x = QDense(10, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense3")(x)
x_out = x
x = Activation("softmax", name="softmax")(x)


model = Model(inputs=[x_in], outputs=[x])
mo = Model(inputs=[x_in], outputs=[x_out])
model.summary()

model.compile(
    loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])



model.save('VGGNetCIFARQuantized2bit.h5')

if train:

  history = model.fit(
      x_train, y_train, batch_size=BATCH_SIZE,
      epochs=NB_EPOCH, initial_epoch=1, verbose=VERBOSE,
      validation_split=VALIDATION_SPLIT)

  outputs = []
  output_names = []

  for layer in model.layers:
    if layer.__class__.__name__ in ["QActivation", "Activation",
                                  "QDense", "QConv2D", "QDepthwiseConv2D"]:
      output_names.append(layer.name)
      outputs.append(layer.output)

  model_debug = Model(inputs=[x_in], outputs=outputs)

  outputs = model_debug.predict(x_train)

  print("{:30} {: 8.4f} {: 8.4f}".format(
      "input", np.min(x_train), np.max(x_train)))

  for n, p in zip(output_names, outputs):
    print("{:30} {: 8.4f} {: 8.4f}".format(n, np.min(p), np.max(p)), end="")
    layer = model.get_layer(n)
    for i, weights in enumerate(layer.get_weights()):
      weights = K.eval(layer.get_quantizers()[i](K.constant(weights)))
      print(" ({: 8.4f} {: 8.4f})".format(np.min(weights), np.max(weights)),
            end="")
      print("")

  p_test = mo.predict(x_test)
  p_test.tofile("p_test.bin")

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


for layer in model.layers:
  for w, weight in enumerate(layer.get_weights()):
    print(layer.name, w, weight.shape)

print_qstats(model)
quantizedweights=model_save_quantized_weights(model)
model_save_quantized_weights(model)
import csv
score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])
my_dict = quantizedweights
np.save('quantizedweightsVGGNetcifar10.npy', my_dict) 

# with open('quantizedweights.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
#     w = csv.DictWriter(f, my_dict.keys())
#     w.writeheader()
#     w.writerow(my_dict)

with open('quantizedweights1.csv','w') as f:
    w = csv.writer(f)
    w.writerow(my_dict.keys())
    w.writerow(my_dict.values())

def hybrid_model():
  """hybrid model that mixes qkeras and keras layers."""

  
  
# x = x_in = Input(
#     x_train.shape[1:-1] + (1,), name="input")
  x = x_in = Input(x_train.shape[1:], name="input")

  x = QConv2D(
    64, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_0_m")(x)
  x = QActivation("quantized_relu(2,0)", name="act0_m")(x)

#Layer2
  x = QConv2D(
    64, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_1_m")(x)
  x = QActivation("quantized_relu(2,0)", name="act1_m")(x)
  x= MaxPool2D(pool_size=(2,2),strides=(2,2),data_format='channels_last')(x)

#Layer3

  x = QConv2D(
    128, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_2_m")(x)
  x = QActivation("quantized_relu(2,0)", name="act2_m")(x)

#Layer4

  x = QConv2D(
    128, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_3_m")(x)
  x = QActivation("quantized_relu(2,0)", name="act3_m")(x)
  x= MaxPool2D(pool_size=(2,2),strides=(2,2),data_format='channels_last')(x)


#Layer5
  x = QConv2D(
    256, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_4_m")(x)
  x = QActivation("quantized_relu(2,0)", name="act4_m")(x)

  x = QConv2D(
    256, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_5_m")(x)
  x = QActivation("quantized_relu(2,0)", name="act5_m")(x)

  x = QConv2D(
    256, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_6_m")(x)
  x = QActivation("quantized_relu(2,0)", name="act6_m")(x)
  x= MaxPool2D(pool_size=(2,2),strides=(2,2),data_format='channels_last')(x)

  x = QConv2D(
    512, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_7_m")(x)
  x = QActivation("quantized_relu(2,0)", name="act7_m")(x)

  x = QConv2D(
    512, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_8_m")(x)
  x = QActivation("quantized_relu(2,0)", name="act8_m")(x)

  x = QConv2D(
    512, (3, 3),padding='same',
    kernel_quantizer=quantized_bits(2,0,1),
    bias_quantizer=quantized_bits(2,0,1),
    name="conv2d_9_m")(x)
  x = QActivation("quantized_relu(2,0)", name="act9_m")(x)
  x=  MaxPool2D(pool_size=(2,2),strides=(2,2),data_format='channels_last')(x)



  x = Flatten()(x)
  x = QDense(4096, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense1")(x)
  x = QActivation("quantized_relu(2,0)", name="act10_m")(x)

  x = QDense(4096, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense2")(x)
  x = QActivation("quantized_relu(2,0)", name="act11_m")(x)

  x = QDense(10, kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense3")(x)
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
