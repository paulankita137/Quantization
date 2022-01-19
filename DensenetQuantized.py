
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""
Created on Mon Sep 13 09:40:26 2021

@author: ANKITA PAUL
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 08:27:47 2021

@author: ANKITA PAUL
"""

"""DenseNet models for Keras.
# Reference paper
- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)
"""


import os

from tensorflow.python.keras import backend as K
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import *

#from tensorflow.python.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
#from tensorflow.keras.engine.network import get_source_inputs
# import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D,ZeroPadding2D, \
    BatchNormalization, concatenate, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import tf_export
from qkeras import QActivation
from qkeras import QDense
from qkeras import quantizers
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings
import argparse
from qkeras import *
from qkeras.utils import model_save_quantized_weights
from tensorflow.keras.datasets import cifar10
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.layers import Concatenate 

DENSENET121_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET121_WEIGHT_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
DENSENET169_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet169_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET169_WEIGHT_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
DENSENET201_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet201_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET201_WEIGHT_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'



np.random.seed(42)

NB_EPOCH = 10
BATCH_SIZE = 500
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
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

x_train /= 255.0
x_test /= 255.0

print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

print(y_train[0:10])

y_train = to_categorical(y_train, NB_CLASSES)
y_test = to_categorical(y_test, NB_CLASSES)

# x = x_in = Input(
#     x_train.shape[1:-1] + (1,), name="input")

def dense_block(x, blocks, name):
  """A dense block.
  Arguments:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.
  Returns:
      output tensor for the block.
  """
  for i in range(blocks):
    x = conv_block(x, 32, name=name + '_block' + str(i + 1))
  return x


def transition_block(x, reduction, name):
  """A transition block.
  Arguments:
      x: input tensor.
      reduction: float, compression rate at transition layers.
      name: string, block label.
  Returns:
      output tensor for the block.
  """
  bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
  x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
  x = QActivation("quantized_relu(2,0)")(x)
  x = QConv2D(
      int(K.int_shape(x)[bn_axis] * reduction),
      1,
      use_bias=False,kernel_quantizer=quantized_bits(2,0,1),
      name=name + '_conv')(
          x)
  x = QAveragePooling2D(2, strides=2, name=name + '_pool')(x)
  return x


def conv_block(x, growth_rate, name):
  """A building block for a dense block.
  Arguments:
      x: input tensor.
      growth_rate: float, growth rate at dense layers.
      name: string, block label.
  Returns:
      output tensor for the block.
  """
  bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
  x1 = BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
          x)
  x1 = QActivation("quantized_relu(2,0)")(x1)
  x1 = QConv2D(4 * growth_rate, 1, use_bias=False,kernel_quantizer=quantized_bits(2,0,1), name=name + '_1_conv')(x1)
  x1 = BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
          x1)
  x1 = QActivation("quantized_relu(2,0)")(x1)
  x1 = QConv2D(
      growth_rate, 3, padding='same', use_bias=False,kernel_quantizer=quantized_bits(2,0,1), name=name + '_2_conv')(
          x1)
      
  x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
  return x


def DenseNet(blocks,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=10):
  """Instantiates the DenseNet architecture.
  Optionally loads weights pre-trained
  on ImageNet. Note that when using TensorFlow,
  for best performance you should set
  `image_data_format='channels_last'` in your Keras config
  at ~/.keras/keras.json.
  The model and the weights are compatible with
  TensorFlow, Theano, and CNTK. The data format
  convention used by the model is the one
  specified in your Keras config file.
  Arguments:
      blocks: numbers of building blocks for the four dense layers.
      include_top: whether to include the fully-connected
          layer at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(224, 224, 3)` (with `channels_last` data format)
          or `(3, 224, 224)` (with `channels_first` data format).
          It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.
  Returns:
      A Keras model instance.
  Raises:
      ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
  """
  if not (weights in {'imagenet', None} or os.path.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as imagenet with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = _obtain_input_shape(
      input_shape,
      default_size=32,
      min_size=32,
      data_format=K.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = Input(shape=input_shape)
  else:
    if not K.is_keras_tensor(input_tensor):
      img_input = Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

  x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
  x = QConv2D(64, 7, strides=2, use_bias=False,kernel_quantizer=quantized_bits(2,0,1), name='conv1/conv')(x)
  x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
  x = QActivation("quantized_relu(2,0)")(x)
  x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
  x = MaxPooling2D(3, strides=2, name='pool1')(x)

  x = dense_block(x, blocks[0], name='conv2')
  x = transition_block(x, 0.5, name='pool2')
  x = dense_block(x, blocks[1], name='conv3')
  x = transition_block(x, 0.5, name='pool3')
  x = dense_block(x, blocks[2], name='conv4')
  x = transition_block(x, 0.5, name='pool4')
  x = dense_block(x, blocks[3], name='conv5')

  x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)

  if include_top:
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    #x = QDense(10, activation='softmax', name='fc1000')(x)
    x= QDense(10,kernel_quantizer=quantized_bits(2,0,1),
           bias_quantizer=quantized_bits(2,0,1),
           name="dense2")(x)
    
    x_out = x
    x = Activation("softmax", name="softmax")(x)
  else:
    if pooling == 'avg':
      x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
      x = GlobalMaxPooling2D(name='max_pool')(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = get_source_inputs(input_tensor)
  else:
    inputs = img_input

  # Create model.
  if blocks == [6, 12, 24, 16]:
    model = Model(inputs, x, name='densenet121')
  elif blocks == [6, 12, 32, 32]:
    model = Model(inputs, x, name='densenet169')
  elif blocks == [6, 12, 48, 32]:
    model = Model(inputs, x, name='densenet201')
  else:
    model = Model(inputs, x, name='densenet')

  # Load weights.
  if weights == 'imagenet':
    if include_top:
      if blocks == [6, 12, 24, 16]:
        weights_path = get_file(
            'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
            DENSENET121_WEIGHT_PATH,
            cache_subdir='models',
            file_hash='0962ca643bae20f9b6771cb844dca3b0')
      elif blocks == [6, 12, 32, 32]:
        weights_path = get_file(
            'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
            DENSENET169_WEIGHT_PATH,
            cache_subdir='models',
            file_hash='bcf9965cf5064a5f9eb6d7dc69386f43')
      elif blocks == [6, 12, 48, 32]:
        weights_path = get_file(
            'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
            DENSENET201_WEIGHT_PATH,
            cache_subdir='models',
            file_hash='7bb75edd58cb43163be7e0005fbe95ef')
    else:
      if blocks == [6, 12, 24, 16]:
        weights_path = get_file(
            'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
            DENSENET121_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='4912a53fbd2a69346e7f2c0b5ec8c6d3')
      elif blocks == [6, 12, 32, 32]:
        weights_path = get_file(
            'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
            DENSENET169_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='50662582284e4cf834ce40ab4dfa58c6')
      elif blocks == [6, 12, 48, 32]:
        weights_path = get_file(
            'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
            DENSENET201_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='1c2de60ee40562448dbac34a0737e798')
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


@tf_export('keras.applications.DenseNet121',
           'keras.applications.densenet.DenseNet121')
def DenseNet121(include_top=True,
                #weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=10):
  return DenseNet([6, 12, 24, 16], include_top, #weights, 
                  input_tensor,
                  input_shape, pooling, classes)


@tf_export('keras.applications.DenseNet169',
           'keras.applications.densenet.DenseNet169')
def DenseNet169(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
  return DenseNet([6, 12, 32, 32], include_top, weights, input_tensor,
                  input_shape, pooling, classes)


@tf_export('keras.applications.DenseNet201',
           'keras.applications.densenet.DenseNet201')
def DenseNet201(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
  return DenseNet([6, 12, 48, 32], include_top, weights, input_tensor,
                  input_shape, pooling, classes)


@tf_export('keras.applications.densenet.preprocess_input')
def preprocess_input(x, data_format=None):
  """Preprocesses a numpy array encoding a batch of images.
  Arguments:
      x: a 3D or 4D numpy array consists of RGB values within [0, 255].
      data_format: data format of the image tensor.
  Returns:
      Preprocessed array.
  """
  return imagenet_utils.preprocess_input(x, data_format, mode='torch')


setattr(DenseNet121, '__doc__', DenseNet.__doc__)
setattr(DenseNet169, '__doc__', DenseNet.__doc__)
setattr(DenseNet201, '__doc__', DenseNet.__doc__)

model=DenseNet121(include_top=('False'))
model.compile(
    loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])


model.summary()
history = model.fit(
      x_train, y_train, batch_size=BATCH_SIZE,
      epochs=NB_EPOCH, initial_epoch=1, verbose=VERBOSE,
      validation_split=VALIDATION_SPLIT)

score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])




model_json = model.to_json()
with open("densenet.json", "w") as json_file:
    json_file.write(model_json)

dense121yo=model_save_quantized_weights(model)

import pickle
# create a binary pickle file 
filezz = open("dense121yo.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(dense121yo,filezz)




#Weight Transitions

import pickle
with open('dense121yo.pkl', 'rb') as fp:
     yo = pickle.load(fp)






from random import seed
from random import choice
neuron_id=[]

seed_list=[1,2,3,4,5,6,7,8,9,10,20,30,50,100,1000]
seed_selection = choice(seed_list)
seed(seed_selection)


z_sequence=[dense121yo['conv1/conv'],dense121yo['conv2_block2_2_conv'],dense121yo['conv2_block5_1_conv'],dense121yo['conv2_block3_1_conv'],dense121yo['conv2_block6_1_conv'],dense121yo['conv3_block6_1_conv'],dense121yo['conv5_block15_1_conv'],dense121yo['pool4_conv'],dense121yo['conv5_block7_1_conv'],dense121yo['conv3_block10_1_conv'],dense121yo['dense2']]

layer_selection = choice(z_sequence)
indx=z_sequence.index(layer_selection)
print('Layer chosen',indx)

neuron_id.append(indx)
    #layer,indx=pick_layer()
if indx==0:
        #first layer
       accuracies=[]
       new_weights=[]
       picked_neuron=[]
       existing_neuron=[]
       for i in range (0,7):
          for j in range(0,7):
             for m in range(0,3):
                 for n in range(0,64):
                     import pickle
                     import numpy 
                     with open('dense121yo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['conv1/conv']['weights'][0][i][j] 
                           k=yo['conv1/conv']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("denseexisting0.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0,0.125,-0.125]

                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("densepicked0.csv", picked_neuron, delimiter=",")   

                           yo['conv1/conv']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_0_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('denseyo.h5')
                           model.load_weights('denseyo.h5') 
                           yolo=model.get_weights()
                           yolo[0][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                           numpy.savetxt("denseaccuracies0.csv", accuracies, delimiter=",")   

                           new_weights.append(weightyo)
                           print("Test accuracy:", score[1])
            
                     n+=1
                 m+=1
             j+=1       
          i+=1
          
elif indx==1:
        #second layer
       accuracies=[]
       new_weights=[]
       picked_neuron=[]
       existing_neuron=[]
       for i in range (0,3):
          for j in range(0,3):
             for m in range(0,128):
                 for n in range(0,32):
                     import pickle
                     import numpy 
                     with open('dense121yo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['conv2_block2_2_conv']['weights'][0][i][j] 
                           k=yo['conv2_block2_2_conv']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("denseexisting0.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0.0625,0,-0.0625]


                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("densepicked0.csv", picked_neuron, delimiter=",")   


                           yo['conv2_block2_2_conv']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_1_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('denseyo.h5')
                           model.load_weights('denseyo.h5') 
                           yolo=model.get_weights()

                           yolo[14][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                          
                           numpy.savetxt("denseaccuracies0.csv", accuracies, delimiter=",")   

                           new_weights.append(weightyo)
                           print("Test accuracy:", score[1])
            
                     n+=1
                 m+=1
             j+=1       
          i+=1
          
elif indx==2:
        #third layer
       accuracies=[]
       new_weights=[]
       picked_neuron=[]
       existing_neuron=[]
       for i in range (0,1):
          for j in range(0,1):
             for m in range(0,192):
                 for n in range(0,128):
                     import pickle
                     import numpy 
                     with open('dense121yo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['conv2_block5_1_conv']['weights'][0][i][j] 
                           k=yo['conv2_block5_1_conv']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("denseexisting0.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0,0.125,-0.125]
                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("densepicked0.csv", picked_neuron, delimiter=",")   

                           yo['conv2_block5_1_conv']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_2_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('denseyo.h5')
                           model.load_weights('denseyo.h5') 
                           yolo=model.get_weights()

                           yolo[49][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                           numpy.savetxt("denseaccuracies0.csv", accuracies, delimiter=",")   

                           new_weights.append(weightyo)
                           print("Test accuracy:", score[1])
            
                     n+=1
                 m+=1
             j+=1       
          i+=1
          
elif indx==3:
        #third layer
       accuracies=[]
       new_weights=[]
       picked_neuron=[]
       existing_neuron=[]
       for i in range (0,1):
          for j in range(0,1):
             for m in range(0,128):
                 for n in range(0,128):
                     import pickle
                     import numpy
                     with open('dense121yo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['conv2_block3_1_conv']['weights'][0][i][j] 
                           k=yo['conv2_block3_1_conv']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("denseexisting0.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0,0.25,0.125,-0.25,-0.125]
                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("densepicked0.csv", picked_neuron, delimiter=",")   

                           yo['conv2_block3_1_conv']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_3_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('denseyo.h5')
                           model.load_weights('denseyo.h5')
                           yolo=model.get_weights()

                           yolo[74][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                           numpy.savetxt("denseaccuracies.csv", accuracies, delimiter=",")   

                           new_weights.append(weightyo)
                           print("Test accuracy:", score[1])
            
                     n+=1
                 m+=1
             j+=1       
          i+=1
          
     
elif indx==4:
        #third layer
       accuracies=[]
       new_weights=[]
       picked_neuron=[]
       existing_neuron=[]
       for i in range (0,1):
          for j in range(0,1):
             for m in range(0,224):
                 for n in range(0,128):
                     import pickle
                     import numpy
                     with open('dense121yo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['conv2_block6_1_conv']['weights'][0][i][j] 
                           k=yo['conv2_block6_1_conv']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("denseexisting0.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0,0.125,-0.125]
                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("densepicked0.csv", picked_neuron, delimiter=",")   

                           yo['conv2_block6_1_conv']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_4_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('vggyo.h5')
                           model.load_weights('vggyo.h5') 
                           yolo=model.get_weights()

                           yolo[104][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                           numpy.savetxt("denseaccuracies0.csv", accuracies, delimiter=",")   

                           new_weights.append(weightyo)
                           print("Test accuracy:", score[1])
            
                     n+=1
                 m+=1
             j+=1       
          i+=1
         
          
   
elif indx==5:
        #third layer
       accuracies=[]
       new_weights=[]
       picked_neuron=[]
       existing_neuron=[]
       for i in range (0,1):
          for j in range(0,1):
             for m in range(0,288):
                 for n in range(0,128):
                     import pickle
                     import numpy
                     with open('dense121yo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['conv3_block6_1_conv']['weights'][0][i][j] 
                           k=yo['conv3_block6_1_conv']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("denseexisting0.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0,0.125,-0.125]
                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("densepicked0.csv", picked_neuron, delimiter=",")   

                           yo['conv3_block6_1_conv']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_4_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('denseyo.h5')
                           model.load_weights('denseyo.h5') 
                           yolo=model.get_weights()

                           yolo[124][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                           numpy.savetxt("denseaccuracies0.csv", accuracies, delimiter=",")   

                           new_weights.append(weightyo)
                           print("Test accuracy:", score[1])
            
                     n+=1
                 m+=1
             j+=1       
          i+=1
  
   
elif indx==6:
        #third layer
       accuracies=[]
       new_weights=[]
       picked_neuron=[]
       existing_neuron=[]
       for i in range (0,1):
          for j in range(0,1):
             for m in range(0,960):
                 for n in range(0,128):
                     import pickle
                     import numpy
                     with open('dense121yo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['conv5_block15_1_conv']['weights'][0][i][j] 
                           k=yo['conv5_block15_1_conv']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("denseexisting0.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0,0.0625,-0.0625]
                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("densepicked0.csv", picked_neuron, delimiter=",")   

                           yo['conv5_block15_1_conv']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_4_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('denseyo.h5')
                           model.load_weights('denseyo.h5') 
                           yolo=model.get_weights()

                           yolo[419][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                           numpy.savetxt("denseaccuracies0.csv", accuracies, delimiter=",")   

                           new_weights.append(weightyo)
                           print("Test accuracy:", score[1])
            
                     n+=1
                 m+=1
             j+=1       
          i+=1
          
   
elif indx==7:
        #third layer
       accuracies=[]
       new_weights=[]
       picked_neuron=[]
       existing_neuron=[]
       for i in range (0,1):
          for j in range(0,1):
             for m in range(0,1024):
                 for n in range(0,512):
                     import pickle
                     import numpy
                     with open('dense121yo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['pool4_conv']['weights'][0][i][j] 
                           k=yo['pool4_conv']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("denseexisting0.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0,0.0625,-0.0625]
                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("densepicked0.csv", picked_neuron, delimiter=",")   

                           yo['pool4_conv']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_4_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('denseyo.h5')
                           model.load_weights('denseyo.h5') 
                           yolo=model.get_weights()

                           yolo[439][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                           numpy.savetxt("denseaccuracies0.csv", accuracies, delimiter=",")   

                           new_weights.append(weightyo)
                           print("Test accuracy:", score[1])
            
                     n+=1
                 m+=1
             j+=1       
          i+=1          
   
          
   
elif indx==8:
        #third layer
       accuracies=[]
       new_weights=[]
       picked_neuron=[]
       existing_neuron=[]
       for i in range (0,1):
          for j in range(0,1):
             for m in range(0,704):
                 for n in range(0,128):
                     import pickle
                     import numpy
                     with open('dense121yo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['conv5_block7_1_conv']['weights'][0][i][j] 
                           k=yo['conv5_block7_1_conv']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("denseexisting0.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0,0.0625,-0.0625]
                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("densepicked0.csv", picked_neuron, delimiter=",")   

                           yo['conv5_block7_1_conv']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_4_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('vggyo.h5')
                           model.load_weights('vggyo.h5') 
                           yolo=model.get_weights()

                           yolo[504][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                           numpy.savetxt("denseaccuracies0.csv", accuracies, delimiter=",")   

                           new_weights.append(weightyo)
                           print("Test accuracy:", score[1])
            
                     n+=1
                 m+=1
             j+=1       
          i+=1   

   
elif indx==9:
        #third layer
       accuracies=[]
       new_weights=[]
       picked_neuron=[]
       existing_neuron=[]
       for i in range (0,1):
          for j in range(0,1):
             for m in range(0,416):
                 for n in range(0,128):
                     import pickle
                     import numpy
                     with open('dense121yo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['conv3_block10_1_conv']['weights'][0][i][j] 
                           k=yo['conv3_block10_1_conv']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("denseexisting0.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0,0.125,-0.125]
                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("densepicked0.csv", picked_neuron, delimiter=",")   

                           yo['conv3_block10_1_conv']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_4_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('denseyo.h5')
                           model.load_weights('denseyo.h5') 
                           yolo=model.get_weights()

                           yolo[249][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                           numpy.savetxt("denseaccuracies0.csv", accuracies, delimiter=",")   

                           new_weights.append(weightyo)
                           print("Test accuracy:", score[1])
            
                     n+=1
                 m+=1
             j+=1       
          i+=1
    
elif indx==10:
        #third layer
       accuracies=[]
       new_weights=[]
       picked_neuron=[]
       existing_neuron=[]
       for i in range (0,1024):
          for j in range(0,10):
             
              import pickle
              import numpy 
              with open('dense121yo.pkl', 'rb') as fp:
                  yo = pickle.load(fp)
                  yo['dense2']['weights'][0][i][j] 
                  k=yo['dense2']['weights'][0][i][j]
                  existing_neuron.append(k)
                  numpy.savetxt("denseexisting0.csv", existing_neuron, delimiter=",")   

                  y_neuron=[0,0.0625,-0.0625]
                  k=choice(y_neuron)
                  picked_neuron.append(k)
                  numpy.savetxt("densepicked0.csv", picked_neuron, delimiter=",")   

                  yo['dense2']['weights'][0][i][j]=k
                           #print(yo['dense0']['weights'][0][i])
                  weightyo=yo
                  model.save_weights('denseyo.h5')
                  model.load_weights('denseyo.h5') 
                  yolo=model.get_weights()

                  yolo[604][i][j]=k
                  model.set_weights(yolo)
                           #print(yolo[0][i][j])
 
                  score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                  accuracies.append(score[1])
                  numpy.savetxt("denseaccuracies0.csv", accuracies, delimiter=",")   

                  new_weights.append(weightyo)
                  print("Test accuracy:", score[1])
                           
                  j+=1 
          i+=1

                        
#changing weights accuracy calculation
#positive1 calculation

# accuracies=[]
# new_weights=[]
# for i in range (3,83):
#     for j in range(0,10):
#        import pickle
#        with open('dense121yo.pkl', 'rb') as fp:
#             yo = pickle.load(fp)
#        yo['dense2']['weights'][0][i][j] 
#        k=yo['dense2']['weights'][0][i][j]
#        if k==0.0:
#            k=0.0625
#            yo['dense2']['weights'][0][i][j]=k
#            print(yo['dense2']['weights'][0][i])
#            weightyo=yo
#            model.save_weights('dense121weightyo.h5')
#            model.load_weights('dense121weightyo.h5')  
#            score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#            accuracies.append(score[1])
#            new_weights.append(weightyo)
#            print("Test accuracy:", score[1])
           
#        elif k==-0.0625:
#            k=0.0
#            yo['dense2']['weights'][0][i][j]=k
#            print(yo['dense2']['weights'][0][i])
#            weightyo=yo
#            model.save_weights('dense121weightyo.h5')
#            model.load_weights('dense121weightyo.h5')  
#            score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#            accuracies.append(score[1])
#            new_weights.append(weightyo)

#            print("Test accuracy:", score[1]) 
           
#        else:
#            j+=1
#     i+=1

# score = model.evaluate(x_test, y_test, verbose=VERBOSE)
# print("Test score:", score[0])
# print("Test accuracy:", score[1])

# anku=(weightyo['dense2']['weights'][0]).reshape(10240,1)

            
#  #getting all the original weights           
# individualweights=[]
# for i in range(0,83):
#     for j in range(0,10):
        
#         iw= yo['dense3']['weights'][0][i][j]
#         individualweights.append(iw)
#         j+=1
     
#     i+=1   
        
  
        
  
# #Positive 2 Transition
# accuracies=[]
# new_weights=[]
# for i in range (0,83):
#     for j in range(0,10):
#        import pickle
#        with open('dense121yo.pkl', 'rb') as fp:
#             yo = pickle.load(fp)
#        yo['dense2']['weights'][0][i][j] 
#        k=yo['dense2']['weights'][0][i][j]
#        if k==-0.0625:
#            k=0.0625
#            yo['dense2']['weights'][0][i][j]=k
#            print(yo['dense2']['weights'][0][i])
#            weightyo=yo
#            model.save_weights('dense121.h5')
#            model.load_weights('dense121.h5')  
#            score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#            accuracies.append(score[1])
#            new_weights.append(weightyo)
#            print("Test accuracy:", score[1])
       
           
#        else:
#            j+=1
#     i+=1

# score = model.evaluate(x_test, y_test, verbose=VERBOSE)
# print("Test score:", score[0])
# print("Test accuracy:", score[1])




# #negative 1 transition

# #model.load_weights('LeNetCifarQuantized2bit.h5')
# accuracies=[]
# new_weights=[]
# for i in range (0,83):
#     for j in range(0,10):
#        import pickle
#        with open('dense121yo.pkl', 'rb') as fp:
#             yo = pickle.load(fp)
#        yo['dense2']['weights'][0][i][j] 
#        k=yo['dense2']['weights'][0][i][j]
#        if k==0.0:
#            k=-0.0625
#            yo['dense2']['weights'][0][i][j]=k
#            print(yo['dense2']['weights'][0][i])
#            weightyo=yo
#            model.save_weights('dense121weightyo.h5')
#            model.load_weights('dense121weightyo.h5')  
#            score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#            accuracies.append(score[1])
#            new_weights.append(weightyo)
#            print("Test accuracy:", score[1])
           
#        elif k==0.0625:
#            k=0.0
#            yo['dense2']['weights'][0][i][j]=k
#            print(yo['dense2']['weights'][0][i])
#            weightyo=yo
#            model.save_weights('dense121weightyo.h5')
#            model.load_weights('dense121weightyo.h5')  
#            score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#            accuracies.append(score[1])
#            new_weights.append(weightyo)

#            print("Test accuracy:", score[1]) 
           
#        else:
#            j+=1
#     i+=1

# score = model.evaluate(x_test, y_test, verbose=VERBOSE)
# print("Test score:", score[0])
# print("Test accuracy:", score[1])








# #Negative 2 Transition

# accuracies=[]
# new_weights=[]
# for i in range (0,83):
#     for j in range(0,10):
#        import pickle
#        with open('dense121yo.pkl', 'rb') as fp:
#             yo = pickle.load(fp)
#        yo['dense2']['weights'][0][i][j] 
#        k=yo['dense2']['weights'][0][i][j]
#        if k==0.0625:
#            k=-0.0625
#            yo['dense2']['weights'][0][i][j]=k
#            print(yo['dense2']['weights'][0][i])
#            weightyo=yo
#            model.save_weights('weightyo.h5')
#            model.load_weights('weightyo.h5')  
#            score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#            accuracies.append(score[1])
#            new_weights.append(weightyo)
#            print("Test accuracy:", score[1])
       
           
#        else:
#            j+=1
#     i+=1

# score = model.evaluate(x_test, y_test, verbose=VERBOSE)
# print("Test score:", score[0])
# print("Test accuracy:", score[1])

# #First Layer Weight Change

# #Get all original weights

# individualweights=[]
# for i in range(0,83):
#     for j in range(0,10):
        
#         iw= yo['conv1/conv']['weights'][0][i][j]
#         individualweights.append(iw)
#         j+=1
     
#     i+=1   
    
    
    
# individualweights=[]
  
# for i in range (0,7):
#     for j in range(0,7):
#        for m in range(0,3):
#          for n in range(0,64):
#              import pickle
#              with open('dense121yo.pkl', 'rb') as fp:
#                    yo = pickle.load(fp)
#              yo['conv1/conv']['weights'][0][i][j] 
#              iw= yo['conv1/conv']['weights'][0][i][j][m][n]
#              individualweights.append(iw)
#              n+=1
#          m+=1
#        j+=1
#     i+=1
             
             
             
#  #positive 1   
# accuracies=[]
# new_weights=[]
# for i in range (0,7):
#     for j in range(0,7):
#        for m in range(0,3):
#          for n in range(0,64):
#              import pickle
#              with open('dense121yo.pkl', 'rb') as fp:
#                    yo = pickle.load(fp)
#              yo['conv1/conv']['weights'][0][i][j] 
#              k=yo['conv1/conv']['weights'][0][i][j][m][n]
             
#              if k==-0.125:
#               k=0.125
#               yo['conv1/conv']['weights'][0][i][j][m][n]=k
#               print(yo['conv1/conv']['weights'][0][i][j][m])
#               weightyo=yo
#               model.save_weights('densenet121weightyo.h5')
#               model.load_weights('densenet121weightyo.h5')  
#               score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#               accuracies.append(score[1])
#               new_weights.append(weightyo)
#               print("Test accuracy:", score[1])
#              elif k==0.0:
#                 k==0.125
#                 yo['conv1/conv']['weights'][0][i][j][m][n]=k
#                 print(yo['conv1/conv']['weights'][0][i][j][m])
#                 weightyo=yo
#                 model.save_weights('densenet121weightyo.h5')
#                 model.load_weights('densenet121weightyo.h5')  
#                 score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#                 accuracies.append(score[1])
#                 new_weights.append(weightyo)
#                 print("Test accuracy:", score[1])
                 
#              else:
#                n+=1
#        m+=1
#        j+=1       
#     i+=1

# score = model.evaluate(x_test, y_test, verbose=VERBOSE)
# print("Test score:", score[0])
# print("Test accuracy:", score[1])



# #Negative 2

# accuracies=[]
# new_weights=[]
# for i in range (0,7):
#     for j in range(0,7):
#        for m in range(0,3):
#          for n in range(0,64):
#              import pickle
#              with open('dense121yo.pkl', 'rb') as fp:
#                    yo = pickle.load(fp)
#              yo['conv1/conv']['weights'][0][i][j] 
#              k=yo['conv1/conv']['weights'][0][i][j][m][n]
             
#              if k==0.125:
#               k=-0.125
#               yo['conv1/conv']['weights'][0][i][j][m][n]=k
#               print(yo['conv1/conv']['weights'][0][i][j][m])
#               weightyo=yo
#               model.save_weights('densenet121weightyo.h5')
#               model.load_weights('densenet121weightyo.h5')  
#               score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#               accuracies.append(score[1])
#               new_weights.append(weightyo)
#               print("Test accuracy:", score[1])
           
#              else:
#                n+=1
#        m+=1
#        j+=1       
#     i+=1

# score = model.evaluate(x_test, y_test, verbose=VERBOSE)
# print("Test score:", score[0])
# print("Test accuracy:", score[1])



# #positive 2
# accuracies=[]
# new_weights=[]
# for i in range (0,7):
#     for j in range(0,7):
#        for m in range(0,3):
#          for n in range(0,64):
#              import pickle
#              with open('dense121yo.pkl', 'rb') as fp:
#                    yo = pickle.load(fp)
#              yo['conv1/conv']['weights'][0][i][j] 
#              k=yo['conv1/conv']['weights'][0][i][j][m][n]
             
#              if k==-0.125:
#               k=0.125
#               yo['conv1/conv']['weights'][0][i][j][m][n]=k
#               print(yo['conv1/conv']['weights'][0][i][j][m])
#               weightyo=yo
#               model.save_weights('densenet121weightyo.h5')
#               model.load_weights('densenet121weightyo.h5')  
#               score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#               accuracies.append(score[1])
#               new_weights.append(weightyo)
#               print("Test accuracy:", score[1])
           
#              else:
#                n+=1
#        m+=1
#        j+=1       
#     i+=1

# score = model.evaluate(x_test, y_test, verbose=VERBOSE)
# print("Test score:", score[0])
# print("Test accuracy:", score[1])





# import pandas as pd
# dataset=pd.read_csv('automate.csv')
# X=dataset.iloc[0:].values
# automate=[]
# for i in range(0,830):
#     if X[i]==0.0:
        
#         automate.append(0.4831)
#     elif X[i]==0.25:
        
#         automate.append(0.4831)
#     else :
#         automate.append(1)
    
#     i+=1





# #middle layer weight change


# individualweights=[]
  
# for i in range (0,1):
#     for j in range(0,1):
#        for m in range(0,832):
#          for n in range(0,128):
#              import pickle
#              with open('dense121yo.pkl', 'rb') as fp:
#                    yo = pickle.load(fp)
#              yo['conv4_block19_1_conv']['weights'][0][i][j] 
#              iw= yo['conv4_block19_1_conv']['weights'][0][i][j][m][n]
#              individualweights.append(iw)
#              n+=1
#          m+=1
#        j+=1
#     i+=1
       


# import numpy
# numpy.savetxt("densenetmiddleblockweights.csv", individualweights, delimiter=",")   


#  #positive 1   
# accuracies=[]
# new_weights=[]
# for i in range (0,1):
#     for j in range(0,1):
#        for m in range(0,832):
#          for n in range(0,128):
#              import pickle
#              with open('dense121yo.pkl', 'rb') as fp:
#                    yo = pickle.load(fp)
#              yo['conv4_block19_1_conv']['weights'][0][i][j] 
#              k=yo['conv4_block19_1_conv']['weights'][0][i][j][m][n]
             
#              if k==-0.125:
#               k=0.125
#               yo['conv4_block19_1_conv']['weights'][0][i][j][m][n]=k
#               print(yo['conv4_block19_1_conv']['weights'][0][i][j][m])
#               weightyo=yo
#               model.save_weights('densenet121midweightyo.h5')
#               model.load_weights('densenet121midweightyo.h5')  
#               score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#               accuracies.append(score[1])
#               new_weights.append(weightyo)
#               print("Test accuracy:", score[1])
#              elif k==0.0:
#                 k==0.125
#                 yo['conv4_block19_1_conv']['weights'][0][i][j][m][n]=k
#                 print(yo['conv4_block19_1_conv']['weights'][0][i][j][m])
#                 weightyo=yo
#                 model.save_weights('densenet121midweightyo.h5')
#                 model.load_weights('densenet121midweightyo.h5')  
#                 score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#                 accuracies.append(score[1])
#                 new_weights.append(weightyo)
#                 print("Test accuracy:", score[1])
                 
#              else:
#                n+=1
#        m+=1
#        j+=1       
#     i+=1

# score = model.evaluate(x_test, y_test, verbose=VERBOSE)
# print("Test score:", score[0])
# print("Test accuracy:", score[1])



# #Negative 2

# accuracies=[]
# new_weights=[]
# for i in range (0,7):
#     for j in range(0,7):
#        for m in range(0,3):
#          for n in range(0,64):
#              import pickle
#              with open('dense121yo.pkl', 'rb') as fp:
#                    yo = pickle.load(fp)
#              yo['conv1/conv']['weights'][0][i][j] 
#              k=yo['conv1/conv']['weights'][0][i][j][m][n]
             
#              if k==0.125:
#               k=-0.125
#               yo['conv1/conv']['weights'][0][i][j][m][n]=k
#               print(yo['conv1/conv']['weights'][0][i][j][m])
#               weightyo=yo
#               model.save_weights('densenet121weightyo.h5')
#               model.load_weights('densenet121weightyo.h5')  
#               score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#               accuracies.append(score[1])
#               new_weights.append(weightyo)
#               print("Test accuracy:", score[1])
           
#              else:
#                n+=1
#        m+=1
#        j+=1       
#     i+=1

# score = model.evaluate(x_test, y_test, verbose=VERBOSE)
# print("Test score:", score[0])
# print("Test accuracy:", score[1])



# #positive 2
# accuracies=[]
# new_weights=[]
# for i in range (0,1):
#     for j in range(0,1):
#        for m in range(0,832):
#          for n in range(0,128):
#              import pickle
#              with open('dense121yo.pkl', 'rb') as fp:
#                    yo = pickle.load(fp)
#              yo['conv4_block19_1_conv']['weights'][0][i][j] 
#              k=yo['conv4_block19_1_conv']['weights'][0][i][j][m][n]
             
#              if k==-0.0625:
#               k=0.0625
#               yo['conv4_block19_1_conv']['weights'][0][i][j][m][n]=k
#               print(yo['conv4_block19_1_conv']['weights'][0][i][j][m])
#               weightyo=yo
#               model.save_weights('densenet121weightyo.h5')
#               model.load_weights('densenet121weightyo.h5')  
#               score = model.evaluate(x_test, y_test, verbose=VERBOSE)
#               accuracies.append(score[1])
#               new_weights.append(weightyo)
#               print("Test accuracy:", score[1])
           
#              else:
#                n+=1
#        m+=1
#        j+=1       
#     i+=1

# score = model.evaluate(x_test, y_test, verbose=VERBOSE)
# print("Test score:", score[0])
# print("Test accuracy:", score[1])





# import pandas as pd
# dataset=pd.read_csv('automate.csv')
# X=dataset.iloc[0:].values
# automate=[]
# for i in range(0,830):
#     if X[i]==0.0:
        
#         automate.append(0.4831)
#     elif X[i]==0.25:
        
#         automate.append(0.4831)
#     else :
#         automate.append(1)
    
#     i+=1







