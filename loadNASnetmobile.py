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
from tensorflow.keras.applications import NASNetMobile

"""
======================================================
============== Load the NASnetmobile model ===========
======================================================
It is ncessary to define the whole model architecture
as it was during training since we are loading the model
from the saved weights
"""

def NASnetmobile_model():
    cl_image_size = 224
    inputs = Input(shape=(cl_image_size,cl_image_size, 3), name="input_image")
    #inputs = data_augmentation(inputs)
    
    base_model = NASNetMobile(input_tensor=inputs, 
                             weights=None, 
                             include_top=False)

    base_model.trainable = True
    # Create new model on top
    inputs = keras.Input(shape=(cl_image_size, cl_image_size, 3))
    x = base_model(inputs, training=False)
    # add a global spatial average pooling layer
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout

    # Classification layer or output layer
    outputs = keras.layers.Dense(4, activation='softmax')(x)

    # this is the model we will train
    NASnetmobile_model = keras.Model(inputs, outputs)
    return NASnetmobile_model

