"""
======================================================
========== Import modules and libraries ==============
======================================================
"""
import os
import glob
import skimage
import numpy as np
import matplotlib.image as mpimg

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,UpSampling2D,Activation
from tensorflow.keras.layers import Concatenate,BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from skimage import transform, draw,io
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    #instantiate the mobileNetV2 architecture
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
    
    model = Model(inputs, x)
    return model

"""
======================================================
============== Define helper functions ===============
======================================================
"""
# lead the image
def read_image(path):
    x = mpimg.imread(path)
    return x
# generate black and white mask from the prediction results
def mask_parse(mask,Image_orig):
    print('generating mask')
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))# change from BGR to RGB

    # create a black boundary around the mask to avoid some objects touching the 
    # bounday
    maskb = np.zeros((256, 256,3), dtype=np.uint8)
    rr, cc = skimage.draw.rectangle((2, 2), (254, 254), shape=maskb.shape)
    maskb[rr, cc] = 1
    image_mask = maskb*mask 
    body_mask = image_mask*255
    full_mask = skimage.transform.resize(body_mask,
                                         Image_orig.shape,
                                         anti_aliasing=True)
    print('mask generated')
    return full_mask

"""
======================================================
============== Evaluate the Unet Model ===============
======================================================
"""
#rootpath = './images_Unet/Evaluation_masks_Nadam/'
rootpath = './images_Unet/Evaluation_masks_Adam/'
#rootpath = './images_Unet/Evaluation_masks_RMSprop/'

""" Load the model weights and start evaluation"""
maskmodel = maskmodel()
#model_path_Nadam="./model_files/Unet_Optimizer_Nadam.h5"
model_path_Adam="./model_files/Unet_Optimizer_Adam.h5"
#model_path_RMSprop="./model_files/Unet_Optimizer_RMSprop.h5"

#maskmodel.load_weights(model_path_Nadam)
maskmodel.load_weights(model_path_Adam)
#maskmodel.load_weights(model_path_RMSprop) 
print('Loaded mask model successfully')

"""Get all files for evaluation"""
evalpath = "./images_Unet/Evaluation_data/"

"""Loop through all images one by one"""

## glob.glob() return a list of file name with specified pathname
for IMG_PATH in glob.glob(r"./images_Unet/Evaluation_data/"+"*.jpg"):

    """Read the image"""
    Image = read_image(IMG_PATH)
    
    """Resize the image and store in different variable"""
    image_resized= skimage.transform.resize(Image, (256, 256,3),anti_aliasing=True)
    
    """generate the body using the U-NET model"""
    mask_pred = maskmodel.predict(np.expand_dims(image_resized, axis=0))[0] > 0.5
    MASK = mask_parse(mask_pred,Image)
    
    """ Save the mask in a folder"""
    head, tail = os.path.split(IMG_PATH)
    filename, file_extension = os.path.splitext(tail)
    savepath = rootpath+filename+'.png'
    skimage.io.imsave(savepath, MASK)
    
print('All masks generated. Check them in the masks folder')



