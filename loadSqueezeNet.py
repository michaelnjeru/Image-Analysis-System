"""
======================================================
========== Import modules and libraries ==============
======================================================
"""
import os
import sys
import shutil
import math
import numpy as np
import tensorflow as tf


import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model

import keras_squeezenet
from keras_squeezenet import SqueezeNet


"""
======================================================
============== Load the SquezeNet model ===========
======================================================
It is ncessary to define the whole model architecture
as it was during training since we are loading the model
from the saved weights
"""
bnmomemtum=0.9 
def fire_module(x,squeeze, expand): 

    y = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu',padding='same')(x) 
    y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y) 
    y1 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=1, activation='relu',padding='same')(y) 
    y1 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y1) 
    y3 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=3, activation='relu',padding='same')(y)  
    y3 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y3) 
    return tf.keras.layers.concatenate([y1, y3])

def SqueezeNet_model():
    # Alway define either the input-tensor or input_shape when calling the Squeezenet
    # module 
    base_model = SqueezeNet(include_top=False,
                            weights=None, 
                            input_shape=(227, 227, 3)
                            ) # Do not include the ImageNet classifier at the top.

    base_model.trainable = False
    # Create new model on top
    inputs = keras.Input(shape=(227, 227, 3))
    x = base_model(inputs, training=False)

    # add an extra fire module to extract a few more features
    x = fire_module(x, squeeze=64, expand=256)
    
    # add a global spatial average pooling layer
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout

    # Classification layer or output layer
    outputs = keras.layers.Dense(4, activation='softmax')(x)

    # this is the model we will train
    SqueezeNet_model= keras.Model(inputs, outputs)
    return SqueezeNet_model