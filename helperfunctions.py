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
"""
======================================================
============== Define helper functions ===============
======================================================
"""
def cameraSnap():
    # The class supports the context manager protocol to make
    # this particularly easy (upon exiting the with statement,
    # the close() method is automatically called):
    with picamera.PiCamera() as camera:
       camera.resolution = (2592, 1944)
       camera.start_preview()
       time.sleep(2)
       timestr = time.strftime("%Y%m%d-%H%M%S")
       # /home/pi/project/images
       path = './images/'+timestr+'.jpg'
       camera.capture(path )
       camera.stop_preview()
    return path
# lead the image
def read_image(path):
  x = mpimg.imread(path)
  return x
# generate black and white mask from the prediction results
def mask_parse(mask,Image):
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
                                         Image.shape,
                                         anti_aliasing=True)
    print('mask generated')
    return full_mask

def plot_comparison(original, mask1, mask_name):
    print('printing image and its mask')
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(mask1, cmap=plt.cm.gray)
    ax2.set_title(mask_name)
    ax2.axis('off')
    plt.show()
def animal_count(Image,MASK,class_names,IMG_PATH,cl_image_size,Classifier):
    print('Analysing image')
    
    # Initialize species data with zeros for each image
    species_data = []# four classes
    impala = 0
    other = 0
    warthog = 0
    zebra = 0
    # convert the maske to gray scale before applying the otsu method
    image_gray = skimage.color.rgb2gray(MASK)

    # apply threshold
    thresh = skimage.filters.threshold_otsu(image_gray)
    binary_mask = image_gray > thresh
    # label image regions
    label_image = skimage.measure.label(binary_mask)

    # **************************************************************************
    # create a figure for plotting the image and bounding boxes
    fig, ax = plt.subplots(figsize=(10, 10))

    # display the original Image
    ax.imshow(Image)


    #===============================================================================
    #                    add bounding box and predictions
    #===============================================================================

    # Initialize empty variables to hold prediction probabilities and class labels
    validProb = []
    valid_idx = []
    validPred_label = []

    # Get each the regions detected and recognize the class of the object
    i = 0
    print('generating regions')
    for region in skimage.measure.regionprops(label_image):
      
      # take regions with large enough areas
      if region.area >= 11000:
        
        # get the dimensions of the bounding box
        minr, minc, maxr, maxc = region.bbox

        # Magnify the bbox with a small percentage
        range_X =abs(minr-maxr)
        range_Y =abs(minc-maxc)
        magnification=0.15
        minr = minr-math.floor(magnification*range_X)
        maxr = maxr+math.floor(magnification*range_X)

        minc = minc-math.floor(magnification*range_Y)
        maxc = maxc+math.floor(magnification*range_Y)

        if minr<0: minr=0
        if minc<0: minc=0

        if maxc>Image.shape[1]: maxc=Image.shape[1]
        if maxr>Image.shape[0]: maxr=Image.shape[0]


        #Bounding box (min_row, min_col, max_row, max_col)
        BBOX = region.bbox
        YY = ( minc,maxc)
        XX = ( minr,maxr) 

        # Extract the region around each animal
        print('extracting a region')
        extracted_animal = Image[XX[0]:XX[1], YY[0]:YY[1]]

        # get the shape of the extracted region
        extracted_shape = extracted_animal.shape

        # resize regions larger then 227x227 and zero padd smaller regions
        if extracted_shape[0]>=cl_image_size or extracted_shape[1]>=cl_image_size:
            # resize the extracted region to same size as the input shape of the model
            extracted_region_resized = skimage.transform.resize(extracted_animal,
                                        (cl_image_size, cl_image_size,3),anti_aliasing=True)
        else:

          old_image_height, old_image_width, channels = extracted_animal.shape

          # create new image of desired size and color (blue) for padding
          new_image_width = cl_image_size
          new_image_height = cl_image_size
          color = (0,0,0)
          extracted_region_resized = np.full(
              (new_image_height,new_image_width, channels),
              color, dtype=np.uint8)
          
          # compute center offset
          x_center = (new_image_width - old_image_width) // 2
          y_center = (new_image_height - old_image_height) // 2

          # copy extracted_animal image into center of result image
          extracted_region_resized[y_center:y_center+old_image_height, 
                x_center:x_center+old_image_width] = extracted_animal
                
      # Since the model is trained on mini-batches, the input is a tensor of 
        # shape [batch_size, image_width, image_height, number_of_channels].
        img = np.expand_dims(extracted_region_resized, axis=0)

        # predict with the loaded model
        S_pred = Classifier.predict(img)
        print('classified the region')
        #get the index of the largest probability and the probability
        idx_of_highest_prob = np.argmax(S_pred, axis=1)

        pred_prob = max(max(S_pred))

        #Get the class name of the Highest prediction probability
        pred_label = class_names[int(idx_of_highest_prob)]

        # print a probability percentage with two decimal places
        pred_percentage = pred_prob*100
        pred_p = "{:.2f}".format(pred_percentage)

        # create a label for the bounding bbox
        bbox_label = pred_label +' '+ str(pred_p )+'%'    
        
        # Consider prediction with probabilities greater tha 50%
        if pred_prob>0.1:
          # draw rectangle around segmented animals
          rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
          
          ax.add_patch(rect)
          # add the species name and percentage probability on top of the bbox
          ax.text(minc, minr, bbox_label, fontsize=15,      color='red')
          validProb.append(pred_prob)
          valid_idx.append(i)
          validPred_label.append(pred_label)
          if pred_label=='impala':
              impala = impala+1
          elif pred_label=='other':
              other = other+1
          elif pred_label=='warthog':
              warthog = warthog+1
          elif pred_label=='zebra':
              zebra = zebra+1
          
          i = i+1
    num_of_animals = i
    animal_labels=validPred_label
    
    impala=str(impala)
    species_data.append(impala.zfill(2))
    
    
    other=str(other)
    species_data.append(other.zfill(2))
    
    warthog=str(warthog)
    species_data.append(warthog.zfill(2))
    
    zebra=str(zebra)
    species_data.append(zebra.zfill(2))
    #print(zebra)
    
    #species_data = [ impala,other,warthog,zebra]# four classes
    #save the figure for further use
    if num_of_animals>0:
        print('saving analysed image')
        rootpath = '/home/pi/project/analysed_images/'
        head, tail = os.path.split(IMG_PATH)
        savepath = rootpath+'analysed'+tail
        savedfig = plt.savefig(savepath, bbox_inches='tight')
    #ax.set_axis_off()
    #plt.tight_layout()
    #plt.show()
    #plt.close()
    return num_of_animals, animal_labels,species_data

