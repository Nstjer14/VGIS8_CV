# -*- coding: utf-8 -*-
"""
Created on Tue May  1 13:24:40 2018

@author: Shaggy
"""
import load_images_2_python
import noiseremover_python as noiserm
import pywt

featureVector = []
dataFrame = load_images_2_python.dataFrame

i = 1
for image in dataFrame.image:
    img_without_noise = noiserm.noiseremover(image,0.1,10)
    print("noise removed")
    equalised_img = noiserm.equalisehistogram(img_without_noise,10)
    print("histogram equalised")
    wavelet_results= pywt.wavedec2(equalised_img,'haar',level=3)
    featureVec = wavelet_results[0].reshape(wavelet_results[0].size)
    print("feature extracted")
    featureVector.append(featureVec)
    print(len(featureVector))
    print("Image ",i, "out of ",dataFrame.image.size,"done")
    i = i + 1

print ("FINISHED")

dataFrame['featureVector'] = featureVector
dataFrame.to_pickle('pythonDatabase')