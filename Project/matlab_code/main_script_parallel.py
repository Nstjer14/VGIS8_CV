# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:43:36 2018

@author: Shaggy
"""

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
import time
import multiprocessing as mp
import os
import timeit


def iris_proc (image):
    img_without_noise = noiserm.noiseremover(image,0.1,10)
  #  print("noise removed")
    equalised_img = noiserm.equalisehistogram(img_without_noise,10)
  #  print("histogram equalised")
    wavelet_results= pywt.wavedec2(equalised_img,'haar',level=3)
    featureVec = wavelet_results[0].reshape(wavelet_results[0].size)
    pid = os.getpid ()
    print("Feature Extracted by: process  id:   {:7d}".format(pid))
    return featureVec

#%% Loading Data
featureVector = []
dataFrame = load_images_2_python.dataFrame
discardList = ['0005left_3-polar.jpg','0014right_2-polar.jpg'] # 
dataFrame = dataFrame[~dataFrame['full_path'].isin(discardList)]
print("The following images have been dropped because the iris localisation was not good enough: ", discardList)



#%% Multi Processing


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    #__spec__ = None
    n_cores = mp.cpu_count()
    pool = mp.Pool(processes=n_cores//2)
    start_time = timeit.default_timer()
    featureVector = pool.map_async(iris_proc,dataFrame.image).get()
    pool.close()
    pool.join()
    print("FINISHED")
    print(timeit.default_timer() - start_time) #94.28003701802831
    dataFrame['featureVector'] = featureVector
    dataFrame.to_pickle('pythonDatabase_para')
    #pool.map_async(time.sleep , range (1,5,1)).get()
#%%
'''
i = 1
for image in dataFrame.image:
    
    featureVec = iris_proc(image)
    featureVector.append(featureVec)
     
    #print(len(featureVector))
    print("Image ",i, "out of ",dataFrame.image.size,"done")
    i = i + 1

print ("FINISHED")
'''
#dataFrame['featureVector'] = featureVector
#dataFrame.to_pickle('pythonDatabase')


#%%
