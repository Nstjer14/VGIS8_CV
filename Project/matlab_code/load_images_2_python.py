# -*- coding: utf-8 -*-
"""
Functionality: loads the folder normalised_iris parses through the images.
It saves the "polar.jpg" images and parses through it's string name to get the labels.

Dependencies: Glob, Pandas, Numpy and OpenCV
"""
#from IPython import get_ipython
#get_ipython().magic('reset -sf')

import glob
import pandas as pd
import numpy as np
import cv2
import os
import platform

filelist = glob.glob('normalised_iris/*.jpg')

iris_normal_list = [i for i in filelist if "polar.jpg" in i]

image_list = []
label_list = []
image_number_list = []
full_path_list = []

for entry in iris_normal_list:
    if (platform.system()=='Windows'):
        folder_img_split = entry.split("\\")
    else:
        folder_img_split = entry.split("/")
    #print(folder_img_split)
    subject_img_split = folder_img_split[1].split("_")
    #print(subject_img_split)
    subject = subject_img_split[0]
    image_number = subject_img_split[1]
    #print(entry)
    img = cv2.imread(entry)
    img = img[:,:,1] # All channels are equal to each other since it is a grayscale image. 
    image_list.append(img)
    label_list.append(subject)
    image_number_list.append(image_number)
    full_path_list.append(folder_img_split[1])


dataFrame = pd.DataFrame({'image':image_list,'label':label_list,'image_number':image_number_list,'full_path':full_path_list})
print("Dataframe created")