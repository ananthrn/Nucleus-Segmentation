
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


# # Preprocessing

# ## Preprocessing script

# In[25]:


import skimage.morphology
from multiprocessing import Pool

newShape = (512,512)

def isColor(img):
    return np.any(img[:,:,1] != img[:,:,2])


from scipy.ndimage import gaussian_filter
from skimage.morphology import reconstruction
from skimage import exposure
from skimage.filters import rank
from skimage.morphology import disk

erosionMap = np.array([
    [1,1,0],
    [1,1,1],
    [0,1,1]
    ])

#flt2107
def brightnessImprover(img):
#     img = exposure.adjust_gamma(img, 2)
    pLower, pUpper = np.percentile(img, (0, 100))
    img = exposure.rescale_intensity(img, in_range=(pLower, pUpper))
    img = exposure.adjust_gamma(img, 2)
    return img

#flt2107
def customResize(img, outputShape, mirror=False):
    if mirror and outputShape[0] > img.shape[0]:
        img = skimage.util.crop(img,((5, 5),))
        return skimage.transform.warp(
            img,
            skimage.transform.AffineTransform(),
            output_shape=outputShape,
            mode='reflect')
    
    return skimage.transform.resize(img,outputShape)

#flt2107
def imgTreatment(img):
    if isColor(img):
        # Invert color images
        # (so blue maps to white after conversion to grayscale)
        img = skimage.color.rgb2hed(img[:,:,:3])        
        img = img[:,:,1]
    else:
        img = skimage.color.rgb2gray(img)
    img = brightnessImprover(img)
    return img



## rleToMask retrieved from https://www.kaggle.com/robertkag/rle-to-mask-converter
def rleToMask(rleString,shape):
    rows,cols = shape[:2]
    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    return img


#flt2107
def handleSample(args,inputFolder=None,outputFolder=None,mirror=True,maskOutputFolder=None):
    ### Retrieve args
    sample,rleStrings = args
    ###
    
    sampleFolder = os.path.join(inputFolder,sample)
    mirrorFlag = mirror

    #### Load, preprocess and save image
    img = skimage.util.img_as_float( imread(os.path.join(sampleFolder,'images',sample+'.png')))
    imgShape = img.shape

    img = imgTreatment(img)
    resizedImg = customResize(img,newShape,mirror=mirrorFlag)
    imsave(os.path.join(outputFolder,sample+'.png'),resizedImg)
    ####

    if not maskOutputFolder:
        return None


    #### Combine and resize masks
    
    if rleStrings == None:
        # Retrieve masks from actual image files
        labelCollection = ImageCollection(os.path.join(sampleFolder,'masks','*'))
    else:
        # Retrieve from RLE strings
        labelCollection = map(lambda s: rleToMask(s,imgShape),rleStrings)

    erosionMap = disk(2)
    concat = np.stack(list(map(lambda im: skimage.morphology.binary_erosion(im,selem=erosionMap), labelCollection)),axis=0)
    
    mask = np.any(concat,axis=0).astype(np.float)

    resizedMask = customResize(mask,newShape,mirror=mirrorFlag)

    imsave(os.path.join(maskOutputFolder,sample+'.png'),resizedMask)        
    ####

    return None


#flt2107
from functools import partial
def transformSamples(inputFolder,outputFolder,maskOutputFolder=None,rleMaskFile=None,mirror=True):   
    samples = sorted(list(os.listdir(inputFolder)))
    
    if rleMaskFile:
        rleDf = pd.read_csv(rleMaskFile)
        g = rleDf.groupby('ImageId')
        rleStrDf = g['EncodedPixels'].apply(list)
        rleStrItems = list(rleStrDf.to_dict().items())
        
        rleStrItemsSorted = sorted(rleStrItems,key=lambda t: t[0])
        _, rleStrings = zip(*rleStrItemsSorted)
    else:
        rleStrings = [None] * len(samples)
    
    kwds={
       'inputFolder':inputFolder,
       'outputFolder':outputFolder,
       'maskOutputFolder':maskOutputFolder,
       'mirror': mirror
    }
    with Pool(processes=8) as pool:
        pool.map(partial(handleSample,**kwds),zip(samples,rleStrings))

    
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    transformSamples('stage1_train','train','trainMask')
    transformSamples('stage1_test','test2','testMask2','stage1_solution.csv',mirror=False)


# In[3]:


## Some examples
#flt2107


import matplotlib.pyplot as plt

samples = ['0d2bf916cc8de90d02f4cd4c23ea79b227dbc45d845b4124ffea380c92d34c8c']
samples += ['00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e']
samples += ['5bda829acd824821bc1f3f6573cf065d364653d5322f033a4af943f7a6170566']
samples += ['1db1cddf28e305c9478519cfac144eee2242183fe59061f1f15487e925e8f5b5']


def pltShow(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

for sample in samples:
    img = imread(os.path.join('train',sample+'.png'))
    pltShow(img)
    img = imread(os.path.join('trainMask',sample+'.png'))
    pltShow(img)


# In[ ]:



