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
from tensorflow.keras.applications import MobileNetV2,NASNetMobile


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg

import skimage
from skimage.transform import resize
from skimage import io,color
from skimage.draw import rectangle
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray, gray2rgb
from skimage.measure import label, regionprops

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

''' pi camera'''
import picamera
import time
import serial
"""
======================================================
============== Load the Unet Model ===================
======================================================
# importing  all the functions defined in loadMASKmodel.py
"""
from loadMASKmodel import *
maskmodel = maskmodel()
model_path_body="./model_files/Unet_Optimizer_Adam.h5"
maskmodel.load_weights(model_path_body)    
print('Loaded mask model successfully')

"""
======================================================
============== Load the CLASSIFIER model ===========
======================================================
It is ncessary to define the whole model architecture
as it was during training since we are loading the model
from the saved weights
"""

''' For MobileNetv2'''
'''------------------------------------------------------------'''
print('Loading MobileNetv2 model')
Classifier = MobileNetV2_model()
print('Loaded MobileNetv2 model successfully')
'''------------------------------------------------------------'''


'''load weights'''
print('Loading MobileNetv2 weights')
model_path_sq = "./model_files/MobileNetV2_optimizer_RMSprop.h5"
Classifier.load_weights(model_path_sq)

'''------------------------------------------------------------'''
print('Loaded MobileNetv2 weights successfully')

"""
======================================================
============== Define helper functions ===============
======================================================
"""
from helperfunctions import *
print('Loaded other functions successfully')
"""
======================================================
Check for presence of an animals, if present, capture and
an image and analyse it
======================================================
"""
import time
#RPi.GPIO module 
import RPi.GPIO as GPIO
#Pin Numbering Declaration:
GPIO.setmode(GPIO.BCM)
#Setting a Pin Mode
GPIO.setup(17, GPIO.IN)
GPIO.setup(27, GPIO.OUT)

# READ THE INPUT FROM THE SENSOR
while True:
    if GPIO.input(17):
        print("Animal detected")
        GPIO.output(27, GPIO.HIGH)
        IMG_PATH = cameraSnap()
        print('Image captured successfully')
        
        Image = read_image(IMG_PATH)
        x= skimage.transform.resize(Image, (256, 256,3),anti_aliasing=True) 
        # generate the body mask using the U-NET model
        y_pred2 = maskmodel.predict(np.expand_dims(x, axis=0))[0] > 0.5
        MASK = mask_parse(y_pred2,Image)
        #plot_comparison(Image, MASK, "mask")
        GPIO.output(27, GPIO.LOW)
        # Define the clasffier input image size
        cl_image_size = 224
        # Define all classes
        class_names = ['bushbuck', 'impala', 'llama', 'warthog', 'waterbuck', 'zebra']
        num_of_animals, animal_labels,species_data = animal_count(Image,MASK,class_names,IMG_PATH,cl_image_size,Classifier)
        print('Number of nimals detected: ', num_of_animals)
        print("Animals labels detected: ", animal_labels)
        print("Species data: ",species_data)
        plt.close
        
        """
        Send the number species count in each of the classes.
        the data is coded as as 8-digit number where each pair of digits represent
        the count of the class. the digit pairs follows the same order established by
        the alphabetical order of the class names
        """
        species_data.append('\n')# add end of line character for tbeam to know end of line
        # using list comprehension
        ttgodata = ''.join([str(elem) for elem in species_data])
        tbeam_bytes = bytes(ttgodata, 'utf-8')
        print(tbeam_bytes)
        #"""
        #"""
        if num_of_animals>1:
            ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
            ser.reset_input_buffer()
            
            while True:
                print('Sending Data to TTGO T-Beam')
                ser.write(tbeam_bytes)
                print('TTGO T-Beam responding')
                t_end = time.time()+5*60
                while time.time()<t_end:
                    
                    line = ser.readline().decode('utf-8').rstrip()
                    print(line)
        #"""
    else:
        print("No animal detected")
    time.sleep(1)
