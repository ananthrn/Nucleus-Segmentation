
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



#flt2107
np.random.seed(12345)
indices = np.random.permutation(imageArr.shape[0])
training_idx, val_idx = indices[:600], indices[600:]
tr_im, val_im = imageArr[training_idx], imageArr[val_idx]
tr_mask, val_mask = maskArr[training_idx], maskArr[val_idx]


# Load weight map

# In[4]:


weightMaps = np.load('weightmap.npy')


# In[30]:


# Show a weight map

# i=0

# import matplotlib.pyplot as plt
# plt.imshow(weightMaps[i])
# plt.colorbar()


# ## Define U-net (model, losses etc)

# In[13]:


import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Flatten,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import tensorflow as tf

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


img_rows = 512
img_cols = 512

smooth = 1.


######

#######

#flt2107
# Adapted from https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
# Onehot version
def dice_coef(y_true_and_w, y_pred_1h):
    truth_mask_1h = K.cast(y_true_and_w > 0,'float32')
    y_true = K.cast(K.argmax(truth_mask_1h),'float32')
    y_pred = K.cast(K.argmax(y_pred_1h),'float32')
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#flt2107
def splitMaskAndWeight(y_true):
    mask = K.cast(y_true > 0,'float32')
    weights = K.sum(y_true,axis=-1)
    return (mask,weights)

#flt2107
def weighted_cross_entropy(y_true_and_w, y_pred):
    truth_mask, weights = splitMaskAndWeight(y_true_and_w)
    return weights*K.categorical_crossentropy(truth_mask,y_pred)
    
    
# Retrieved from Kaggle challenge    
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = np.argmax(y_true_in,axis=-1)
    y_pred = np.argmax(y_pred_in,axis=-1)
    
    
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

# Retrieved from Kaggle challenge
def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)

# Retrieved from Kaggle challenge
def my_iou_metric(label, pred):
    truth_mask, weights = splitMaskAndWeight(label)
    metric_value = tf.py_func(iou_metric_batch, [truth_mask, pred], tf.float32)
    return metric_value
    

hiddenChan = 64

# Adapted from Kaggle challenge
def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(hiddenChan, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(hiddenChan, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(hiddenChan*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(hiddenChan*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(hiddenChan*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(hiddenChan*4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(hiddenChan*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(hiddenChan*8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(hiddenChan*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(hiddenChan*16, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(hiddenChan*8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(hiddenChan*8, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(hiddenChan*4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(hiddenChan*4, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(hiddenChan*2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(hiddenChan*2, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(hiddenChan, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(hiddenChan, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(2, (1, 1), activation='softmax')(conv9)
    
    model = Model(inputs=[inputs], outputs=[conv10])

#     model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-4), loss=weighted_cross_entropy,                  metrics=[dice_coef])#,my_iou_metric])

    return model
  

model = get_unet()
model.summary()


# ## Train

# In[6]:



from keras.utils import to_categorical

#flt2107
# Embed weights and training outputs together
# Workaround to be able to use weight maps on Keras
tr_mask_and_w = weightMaps[training_idx,:,:,np.newaxis] * to_categorical(tr_mask,2)
val_mask_and_w = weightMaps[val_idx,:,:,np.newaxis] * to_categorical(val_mask,2)
tr_mask_and_w.shape


# In[34]:


##########################################
#flt2107
from keras.callbacks import *
from keras.utils import to_categorical

weights_filename = 'unetw_weights.h5'


#flt2107
model.fit(
    x=tr_im,
    y=tr_mask_and_w,
    validation_data=(val_im,val_mask_and_w),
    batch_size=8,
    shuffle=True,
    epochs=20,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=3),
        ModelCheckpoint(weights_filename, monitor='val_loss', save_best_only=True, save_weights_only=True),
    ]
)

##########################################


# In[37]:


# fine tuning

weights_filename = 'unetw_weights2.h5'

model.fit(
    x=tr_im,
    y=tr_mask_and_w,
    validation_data=(val_im,val_mask_and_w),
    batch_size=8,
    shuffle=True,
    initial_epoch=20,
    epochs=40,
    callbacks=[
        EarlyStopping(monitor='val_my_iou_metric', patience=3,mode='max'),
        ModelCheckpoint(weights_filename, monitor='val_my_iou_metric', save_best_only=True, save_weights_only=True, mode='max'),
    ]
)


# In[8]:


# More fine tuning

weights_filename = 'unetw_weights3.h5'

from keras.callbacks import *

# Exp decay adapted from version found on Stack overflow
def exp_decay(t,lr):
    initial_lrate = 1e-4
    k = 0.2
    lrate = initial_lrate * np.exp(-k*t)
    return lrate
lrate = LearningRateScheduler(exp_decay,verbose=1)

#flt2107
model.fit(
    x=tr_im,
    y=tr_mask_and_w,
    validation_data=(val_im,val_mask_and_w),
    batch_size=12,
    shuffle=True,
    epochs=10,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=2),
        ModelCheckpoint(weights_filename, monitor='val_loss', save_best_only=True, save_weights_only=True),
        lrate
    ]
)


# In[14]:


model.load_weights('unetw_weights3.h5')



# In[42]:


import matplotlib.pyplot as plt
import skimage.morphology

# Show some sample results (this was original run in a Jupyter notbook)


# i=23

# print("Image")
# plt.figure()
# plt.imshow(val_im[i,:,:,0])
# plt.show()
# print("truth mask")
# plt.figure()
# plt.imshow(val_mask[i,:,:,0])
# plt.show()


# print("predicted mask")
# pred = model.predict(val_im[i:i+1])

# img = np.argmax(pred[0],axis=-1)
# # img = skimage.morphology.erosion(img)
# plt.figure()
# plt.imshow(img)
# plt.show()


# In[16]:



# Let's attach the IOU metric to get metrics for test dataset
model.compile(optimizer=Adam(lr=1e-4), loss=weighted_cross_entropy,              metrics=[dice_coef,my_iou_metric])


# In[26]:


# Load test data


testImageCollection = ImageCollection('test2/*.png')
testMaskCollection = ImageCollection('testMask2/*.png')

testImageArr = testImageCollection.concatenate()
testMaskArr = testMaskCollection.concatenate()

# normImageArr = imageArr[:,:,:,:3] / 255.0
testImageArr = testImageArr[:,:,:,np.newaxis] / 65535
testMaskArr = testMaskArr[:,:,:] / 65535

testMaskArrayCat = to_categorical(testMaskArr,2)
testMaskArrayCat.shape


# In[29]:


# Evaluate

model.load_weights('unetw_weights3.h5')
print(model.evaluate(testImageArr,testMaskArrayCat,verbose=1,batch_size=8))

# [0.1739884750201152, 0.7227377643952003, 0.416923084625831]


# In[62]:


print(model.metrics_names)

# ['loss', 'dice_coef', 'my_iou_metric']

