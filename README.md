# Image-Analysis-System
The image analysis system is based on two DCNN models, one for segmentation  and another for calssification. The system outputs an image with labels of each animal in a bounding box. Therefore counting and recognized animals are obtained from the output image.
For SqueezeNet, the keras application do not have model for it. Therefore it is necessary to first install the models from the zipped file in the Raspberry Pi before running the codes.
The training files are provided as Jupyter notebooks for Google Colab. it is expected the training images and labels are in the google drive. you may need to edit the path of the images to match the actual location in the user case.
Other python files, contains 'path' variables which assumes the referenced folders are subfolders of where the .py file is located.
