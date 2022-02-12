"""
======================================================
========== Import modules and libraries ==============
======================================================
"""
import os
import glob
from loadMASKmodel import *
from loadNASnetmobile import *
"""
======================================================
============== Load the Unet Model ===================
======================================================
# importing  all the functions defined in loadMASKmodel.py
"""

maskmodel = maskmodel()
model_path_Nadam="./model_files/Unet_Optimizer_Adam.h5"
maskmodel.load_weights(model_path_Nadam)    
print('Loaded mask model successfully')

"""
======================================================
============== Load the CLASSIFIER model ===========
======================================================
It is ncessary to define the whole model architecture
as it was during training since we are loading the model
from the saved weights
"""

''' For NASnetmobile'''
'''------------------------------------------------------------'''
print('Loading NASnetmobile model')
Classifier = NASnetmobile_model()
print('Loaded NASnetmobile model successfully')
'''------------------------------------------------------------'''


'''load weights'''
print('Loading NASnetmobile weights')
model_path_RMSprop= "./model_files/NASnetmobile_optimizer_RMSprop.h5"
Classifier.load_weights(model_path_RMSprop)
print('Loaded NASnetmobile weights successfully')

"""
======================================================
============== Define helper functions ===============
======================================================
"""
from helperfunctions_Eval import *
print('Loaded other functions successfully')
"""
======================================================
============== Evaluate the Unet Model ===============
======================================================
"""
analysedpath = './analysed_images/NASnetMobile_RMSprop/'
masksavepath = "./analysed_images/masks/"

"""Loop through all images one by one"""

## glob.glob() return a list of file name with specified pathname
# r"/home/pi/project/quickTestImages/"+"*.jpeg"
# r"/home/pi/project/images_Unet/Evaluation_data/"+"*.jpg"
# /home/pi/project/Evaluation_data/group2
for IMG_PATH in glob.glob(r"./Evaluation_data/"+"*.jpg"):
    # Read image
    Image = read_image(IMG_PATH)
    
    # resize the image for Unet use
    im_resized= skimage.transform.resize(Image, (256, 256,3),anti_aliasing=True) 
    
    # generate the body mask using the U-NET model
    y_pred2 = maskmodel.predict(np.expand_dims(im_resized, axis=0))[0] > 0.5
    MASK = mask_parse(y_pred2,Image)
    
    # plot and save a combination of image and mask
    plot_comparison(Image, MASK, "mask",IMG_PATH,masksavepath)
    
    # Define the clasffier input image size
    cl_image_size = 224
    
    # Define all classes
    class_names = ['impala', 'other', 'warthog','zebra']
    
    #Perform classification
    num_of_animals, animal_labels,species_data = animal_count(Image,MASK,class_names,IMG_PATH,cl_image_size,Classifier,analysedpath)
    print('Number of nimals detected: ', num_of_animals)
    print("Animals labels detected: ", animal_labels)
    
print('All images inalysed and saved in the analysed folder')
    
