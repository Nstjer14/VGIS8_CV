# -*- coding: utf-8 -*-
"""
Created on Tue May  1 13:24:40 2018

@author: Shaggy
"""
import load_images_2_python
import noiseremover_python as noiserm
import pywt
import cv2
import matplotlib.pyplot as plt

featureVector = []
dataFrame = load_images_2_python.dataFrame

#dataFrame = dataFrame.iloc[range(224),:]
#dataFrame = dataFrame.drop(dataFrame.index[224]) 3-polar.jpg 0005left
discardList = ['0005left_3-polar.jpg','0014right_2-polar.jpg'] # 
dataFrame = dataFrame[~dataFrame['full_path'].isin(discardList)]
print("The following images have been dropped because the iris localisation was not good enough: ", discardList)
#%%
i = 1
for image in dataFrame.image:
    
    img_without_noise = noiserm.noiseremover(image,0.1,10)
  #  print("noise removed")
    equalised_img = noiserm.equalisehistogram(img_without_noise,10)
  #  print("histogram equalised")
    wavelet_results= pywt.wavedec2(equalised_img,'haar',level=3)
    featureVec = wavelet_results[0].reshape(wavelet_results[0].size)
    print("feature extracted")        
    featureVector.append(featureVec)
     
    #print(len(featureVector))
    print("Image ",i, "out of ",dataFrame.image.size,"done")
    i = i + 1

print ("FINISHED")

dataFrame['featureVector'] = featureVector
dataFrame.to_pickle('pythonDatabase')