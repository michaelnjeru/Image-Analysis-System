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
from tensorflow.keras.applications import MobileNetV2#,NASNetMobile

"""
======================================================
============== Load the Unet Model ===================
======================================================

It is ncessary to define the whole model architecture
as it was during training since we are loading the model
from the saved weights
"""
def maskmodel():
    inputs = Input(shape=(256, 256, 3), name="input_image")
    #inputs = data_augmentation(inputs)
    
    encoder = MobileNetV2(input_tensor=inputs, weights=None, include_top=False, alpha=1)

    #unFreeze the convolutional base
    encoder.trainable = True
    # freeze the first 100 layers
    fine_tune_at = 100
    for layer in encoder.layers[:fine_tune_at]:
      layer.trainable = False
    # Form th UNET architecture
    encoder(inputs)
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    f = [16, 32, 48, 64]
    
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    
    maskmodel = Model(inputs, x)
    return maskmodel



