# -*- coding: utf-8 -*-
"""
Functionality: Does a discrete 2d Haar wavelet decomposition on the polar iris images.

Depencencies: pywt and the script load_images_2_python to load the images
"""

import pywt
import load_images_2_python


dataFrame = load_images_2_python.dataFrame
featureVector = []

for entry in dataFrame.image:
    wavelet_results= pywt.wavedec2(entry,'haar',level=3)
    featureVec = wavelet_results[0].reshape(wavelet_results[0].size)
    featureVector.append(featureVec)

dataFrame['featureVector'] = featureVector
dataFrame.to_pickle('pythonDatabase')