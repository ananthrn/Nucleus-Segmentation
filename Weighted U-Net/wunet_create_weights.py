
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import pandas as pd
import numpy as np
import skimage
import skimage.transform
from skimage.io import *
from skimage.measure import label
import warnings


import scipy.ndimage as ndi
from keras.utils import to_categorical
from multiprocessing import Pool
from functools import partial


print(skimage.__version__)


# # Unet

# ## Load data into batch matrices

# In[2]:


#flt2107
imageCollection = ImageCollection('train/*.png')
maskCollection = ImageCollection('trainMask/*.png')


imageArr = imageCollection.concatenate()
maskArr = maskCollection.concatenate()

# normImageArr = imageArr[:,:,:,:3] / 255.0
imageArr = imageArr[:,:,:,np.newaxis] / 65535
maskArr = maskArr[:,:,:,np.newaxis] / 65535


# In[3]:




neighborhood = np.array([
    [1,1,0],
    [1,1,1],
    [0,1,1]
])


#flt2107
def getWeightForMask(mask,wcMap=None,w0=10,sigma=5):
    labels, n = ndi.label(mask,neighborhood)
    labels = to_categorical(labels,n+1)[:,:,1:]

    if n == 0:
        w = np.zeros(mask.shape)*1.0
        return w

    distancesPerLabel = np.stack([ndi.distance_transform_edt(1-labels[:,:,i]) for i in range(n)])

    sortedDists = np.sort(distancesPerLabel,axis=0)

    d1, d2 = sortedDists[0], sortedDists[1]

    wc = -np.log(wcMap[mask.astype(np.int)])

    w = wc + w0*np.exp( -np.power(d1+d2,2) / (2*(sigma**2)) )
    
    return w

#flt2107
def getWeightMap(masks,w0=10,sigma=5):
    
    if masks.ndim == 4:
            masks = masks[:,:,:,0]

    onesFraction = np.sum(masks)/masks.size

    wcMap = np.array([1-onesFraction,onesFraction])
    
    kwds = { 'wcMap': wcMap, 'w0':w0, 'sigma':sigma }
    with Pool(processes=8) as pool:
        wl = pool.map(partial(getWeightForMask,**kwds),masks)
        
    return np.stack(wl)

import time
t = time.time()
weightMaps = getWeightMap(maskArr)
print(time.time() - t)


# In[29]:


np.save('weightmap.npy',weightMaps)

