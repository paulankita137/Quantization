# -*- coding: utf-8 -*-


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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input,DepthwiseConv2D, Activation, Dropout, GlobalAveragePooling2D,ZeroPadding2D, \
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
#from tensorflow.python.keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras.layers import Concatenate 




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


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
#from tensorflow.keras.utils.vis_utils import plot_model

from tensorflow.keras import backend as K


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)
  



def mobileconv_block(x, blocks,filters, kernel,strides,name):
  """A dense block.
  Arguments:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.
  Returns:
      output tensor for the block.
  """
  for i in range(blocks):
    x = conv_block(x, filters,kernel,strides, name=name + '_block' + str(i + 1))
  return x
  
    

def _conv_block(inputs, filters, kernel, strides,name):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    # x = BatchNormalization(axis=channel_axis)(x)
    # return Activation(relu6)(x)
    
    x = QConv2D(
      filters, kernel, padding='same', use_bias=False,kernel_quantizer=quantized_bits(2,0,1),name=name+'c2d')(
          inputs)
          
    x = BatchNormalization(
       epsilon=1.001e-5)(
          x)
          
    return QActivation("quantized_relu(2,0)")(x)      




def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        alpha: Integer, width multiplier.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Depth
    tchannel = K.int_shape(inputs)[channel_axis] * t
    # Width
    cchannel = int(filters * alpha)

    #x = _conv_block(inputs, tchannel, (1, 1), (1, 1),name='convyo')
    x = mobileconv_block(x, blocks[1],tchannel,(1,1),(1,1), name='convyo')


    x = QDepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1,name='dcd',depthwise_quantizer=quantized_bits(2,0,1),bias_quantizer=quantized_bits(2,0,1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = QActivation("quantized_relu(2,0)")(x)

    x = QConv2D(cchannel, (1, 1), strides=(1, 1),kernel_quantizer=quantized_bits(2,0,1),name='c2d1',bias_quantizer=quantized_bits(2,0,1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        alpha: Integer, width multiplier.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

    return x


def MobileNetv2(input_shape,blocks, k, alpha=1.0):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
        alpha: Integer, width multiplier, better in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4].
    # Returns
        MobileNetv2 model.
    """
    inputs = Input(shape=input_shape)

    first_filters = _make_divisible(32 * alpha, 8)
    #x = _conv_block(inputs, first_filters, (3, 3), strides=(2, 2),name='convyoyo')
    x = mobileconv_block(inputs, blocks[0],first_filters,(1,1),(1,1), name='convyoyo')

    x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)

    if alpha > 1.0:
        last_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_filters = 1280

    #x = _conv_block(x, last_filters, (1, 1), strides=(1, 1),name='convyoyoyo')
    x = mobileconv_block(x, blocks[2],last_filters,(1,1),(1,1), name='convyoyoyo')
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, last_filters))(x)
    #x = Dropout(0.3, name='Dropout')(x)
    x = QConv2D(k, (1, 1),kernel_quantizer=quantized_bits(2,0,1),name='c2d3',bias_quantizer=quantized_bits(2,0,1), padding='same')(x)
    x = QActivation("quantized_bits(2,0,1)")(x)
    x = Activation('softmax', name='softmax')(x)
    output = Reshape((k,))(x)

    model = Model(inputs, output)
    # plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)

    return model


# if __name__ == '__main__':
#     model = MobileNetv2((224, 224, 3), 10, 1.0)
#     print(model.summary())
    
    
model = MobileNetv2((32, 32, 3),(1,1,1,17), 10, 1.0)    
model.compile(
loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

history = model.fit(
      x_train, y_train, batch_size=BATCH_SIZE,
      epochs=5, initial_epoch=1, verbose=VERBOSE,
      validation_split=VALIDATION_SPLIT)

score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print("Test accuracy:", score[1])


mobilenetv2=model_save_quantized_weights(model)

import pickle
# create a binary pickle file 
filezz = open("mobileyo.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(mobilenetv2,filezz)



from random import seed
from random import choice
neuron_id=[]

seed_list=[1,2,3,4,5,6,7,8,9,10,20,30,50,100,1000]
seed_selection = choice(seed_list)
seed(seed_selection)

import pickle
# with open('lenetyo.pkl', 'rb') as fp:
#          lenetyo = pickle.load(fp)
z_sequence=[mobilenetv2['q_conv2d_130'],mobilenetv2['q_conv2d_134'],mobilenetv2['q_conv2d_140'],mobilenetv2['q_conv2d_145'],mobilenetv2['q_conv2d_150'],mobilenetv2['q_conv2d_160'],mobilenetv2['q_conv2d_166']]

layer_selection = choice(z_sequence)
#indx=z_sequence.index(layer_selection)
indx=0
print('Layer chosen',indx)

neuron_id.append(indx)
    #layer,indx=pick_layer()
if indx==0:
        #first layer
       accuracies=[]
       new_weights=[]
       picked_neuron=[]
       existing_neuron=[]
       for i in range (0,3):
          for j in range(0,3):
             for m in range(0,3):
                 for n in range(0,32):
                     import pickle
                     import numpy 
                     import pickle
                      # create a binary pickle file 
                     filezz = open("mobileyo.pkl","wb")

                      # write the python object (dict) to pickle file
                     pickle.dump(mobilenetv2,filezz)

                     with open('mobileyo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['q_conv2d_130']['weights'][0][i][j] 
                           k=yo['q_conv2d_130']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("mobileexisting0.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0,0.25,-0.25]

                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("mobilepicked0.csv", picked_neuron, delimiter=",")   

                           yo['q_conv2d_130']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_0_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('mobileyo.h5')
                           model.load_weights('mobileyo.h5') 
                           yolo=model.get_weights()
                           yolo[0][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                           numpy.savetxt("mobileaccuracies0.csv", accuracies, delimiter=",")   

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
       for i in range (0,1):
          for j in range(0,1):
             for m in range(0,96):
                 for n in range(0,24):
                     import pickle
                     import numpy 
                     import pickle
                      # create a binary pickle file 
                     filezz = open("mobileyo.pkl","wb")

                      # write the python object (dict) to pickle file
                     pickle.dump(mobilenetv2,filezz)
                     with open('mobileyo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['q_conv2d_134']['weights'][0][i][j] 
                           k=yo['q_conv2d_134']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("mobileexisting1.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0.0625,0,-0.0625]


                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("mobilepicked1.csv", picked_neuron, delimiter=",")   


                           yo['q_conv2d_134']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_1_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('mobileyo.h5')
                           model.load_weights('mobileyo.h5') 
                           yolo=model.get_weights()

                           yolo[33][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                          
                           numpy.savetxt("mobileaccuracies1.csv", accuracies, delimiter=",")   

                           new_weights.append(weightyo)
                           print("Test accuracy:", score[1])
            
                     n+=1
                 m+=1
             j+=1       
          i+=1
          
          
          
elif indx==2:
        #second layer
       accuracies=[]
       new_weights=[]
       picked_neuron=[]
       existing_neuron=[]
       for i in range (0,1):
          for j in range(0,1):
             for m in range(0,192):
                 for n in range(0,32):
                     import pickle
                     import numpy 
                     import pickle
                      # create a binary pickle file 
                     filezz = open("mobileyo.pkl","wb")

                      # write the python object (dict) to pickle file
                     pickle.dump(mobilenetv2,filezz)
                     with open('mobileyo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['q_conv2d_140']['weights'][0][i][j] 
                           k=yo['q_conv2d_140']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("mobileexisting2.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0.125,0,-0.125]


                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("mobilepicked2.csv", picked_neuron, delimiter=",")   


                           yo['q_conv2d_140']['weights'][0][i][j][m][n]=k
                           #print(yo['q_conv2d_140']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('mobileyo.h5')
                           model.load_weights('mobileyo.h5') 
                           yolo=model.get_weights()

                           yolo[84][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                          
                           numpy.savetxt("mobileaccuracies2.csv", accuracies, delimiter=",")   

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
             for m in range(0,64):
                 for n in range(0,384):
                     import pickle
                     import numpy 
                     import pickle
                      # create a binary pickle file 
                     filezz = open("mobileyo.pkl","wb")

                      # write the python object (dict) to pickle file
                     pickle.dump(mobilenetv2,filezz)
                     with open('mobileyo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['q_conv2d_145']['weights'][0][i][j] 
                           k=yo['q_conv2d_145']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("mobileexisting3.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0.25,0,-0.25]


                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("mobilepicked3.csv", picked_neuron, delimiter=",")   


                           yo['q_conv2d_145']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_1_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('mobileyo.h5')
                           model.load_weights('mobileyo.h5') 
                           yolo=model.get_weights()

                           yolo[124][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                          
                           numpy.savetxt("mobileaccuracies3.csv", accuracies, delimiter=",")   

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
             for m in range(0,384):
                 for n in range(0,64):
                     import pickle
                     import numpy 
                     import pickle
                      # create a binary pickle file 
                     filezz = open("mobileyo.pkl","wb")

                      # write the python object (dict) to pickle file
                     pickle.dump(mobilenetv2,filezz)
                     with open('mobileyo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['q_conv2d_150']['weights'][0][i][j] 
                           k=yo['q_conv2d_150']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("mobileexisting4.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0.125,0,-0.125]


                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("mobilepicked4.csv", picked_neuron, delimiter=",")   


                           yo['q_conv2d_150']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_1_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('mobileyo.h5')
                           model.load_weights('mobileyo.h5') 
                           yolo=model.get_weights()

                           yolo[135][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                          
                           numpy.savetxt("mobileaccuracies4.csv", accuracies, delimiter=",")   

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
             for m in range(0,96):
                 for n in range(0,576):
                     import pickle
                     import numpy 
                     import pickle
                      # create a binary pickle file 
                     filezz = open("mobileyo.pkl","wb")

                      # write the python object (dict) to pickle file
                     pickle.dump(mobilenetv2,filezz)
                     with open('mobileyo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['q_conv2d_155']['weights'][0][i][j] 
                           k=yo['q_conv2d_155']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("mobileexisting5.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0.25,0,-0.25]


                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("mobilepicked5.csv", picked_neuron, delimiter=",")   


                           yo['q_conv2d_155']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_1_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('mobileyo.h5')
                           model.load_weights('mobileyo.h5') 
                           yolo=model.get_weights()

                           yolo[192][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                          
                           numpy.savetxt("mobileaccuracies5.csv", accuracies, delimiter=",")   

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
                 for n in range(0,160):
                     import pickle
                     import numpy 
                     import pickle
                      # create a binary pickle file 
                     filezz = open("mobileyo.pkl","wb")

                      # write the python object (dict) to pickle file
                     pickle.dump(mobilenetv2,filezz)
                     with open('mobileyo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['q_conv2d_160']['weights'][0][i][j] 
                           k=yo['q_conv2d_160']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("mobileexisting6.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0.0625,0,-0.0625]


                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("mobilepicked6.csv", picked_neuron, delimiter=",")   


                           yo['conv2d_1_m']['weights'][0][i][j][m][n]=k
                           #print(yo['conv2d_1_m']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('mobileyo.h5')
                           model.load_weights('mobileyo.h5') 
                           yolo=model.get_weights()

                           yolo[254][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                          
                           numpy.savetxt("mobileaccuracies6.csv", accuracies, delimiter=",")   

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
             for m in range(0,1280):
                 for n in range(0,10):
                     import pickle
                     import numpy 
                     import pickle
                      # create a binary pickle file 
                     filezz = open("mobileyo.pkl","wb")

                      # write the python object (dict) to pickle file
                     pickle.dump(mobilenetv2,filezz)
                     with open('mobileyo.pkl', 'rb') as fp:
                           yo = pickle.load(fp)
                           yo['q_conv2d_166']['weights'][0][i][j] 
                           k=yo['q_conv2d_166']['weights'][0][i][j][m][n]
                           existing_neuron.append(k)
                           numpy.savetxt("mobileexisting7.csv", existing_neuron, delimiter=",")   

                           y_neuron=[0.0625,0,-0.0625]


                           k=choice(y_neuron)
                           picked_neuron.append(k)
                           numpy.savetxt("mobilepicked7.csv", picked_neuron, delimiter=",")   


                           yo['q_conv2d_166']['weights'][0][i][j][m][n]=k
                           #print(yo['q_conv2d_166']['weights'][0][i][j][m])
                           weightyo=yo
                           model.save_weights('mobileyo.h5')
                           model.load_weights('mobileyo.h5') 
                           yolo=model.get_weights()

                           yolo[299][i][j][m][n]=k
                           model.set_weights(yolo)
                           #print(yolo[0][i][j][m][n])
 
                           score = model.evaluate(x_test, y_test, verbose=VERBOSE)
                           accuracies.append(score[1])
                          
                           numpy.savetxt("mobileaccuracies7.csv", accuracies, delimiter=",")   

                           new_weights.append(weightyo)
                           print("Test accuracy:", score[1])
            
                     n+=1
                 m+=1
             j+=1       
          i+=1                
                    
          
                    
     
             
   
                