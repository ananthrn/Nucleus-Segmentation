# Author : @kr2741 

# Custom Keras generator to generate augmented training images on the fly.
import Augmentor
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from PIL import Image
from scipy.misc import imsave

# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py
def augmented_image_generator(X_train, Y_train, augmentation, batch_size=1):
    while True:
        
        n = len(X_train)
        batch_images = np.zeros((batch_size, 512, 512, 3))
        batch_masks = np.zeros((batch_size, 512, 512))
        
        for i in range(batch_size):
            
            index = random.sample(list(range(n)),1)
            
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                                   "Fliplr", "Flipud", "CropAndPad",
                                   "Affine"]

            def hook(images, augmenter, parents, default):
                return (augmenter.__class__.__name__ in MASK_AUGMENTERS)
            
            shape = (X_train[index].shape[1], X_train[index].shape[2], X_train[index].shape[3])
            image = np.reshape(X_train[index],shape)
            mask = np.squeeze(Y_train[index])

            image_shape = image.shape
            mask_shape = mask.shape
            deterministicAugmentor = augmentation.to_deterministic()
            image = deterministicAugmentor.augment_image(image)
            mask = deterministicAugmentor.augment_image(mask.astype(np.uint8),
                                         hooks=imgaug.HooksImages(activator=hook))
            batch_images[i] = image
            batch_masks[i] = mask
        
        yield batch_images, batch_masks

TRAIN_PATH = '/home/kr2741/data/data-science-bowl-2018/stage1_train/'
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
train_ids = next(os.walk(TRAIN_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
SAVE_PATH = "/home/kr2741/data/data-science-bowl-2018/stage_train_for_augmentation/"

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    
    Y_train[n] = mask
    MASK_DIR = SAVE_PATH+"/masks/"
    IMAGE_DIR = SAVE_PATH+"/images/"
    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    plt.figure()
    plt.imsave(IMAGE_DIR + str(id_) + ".png", X_train[n])
    plt.figure()
    plt.imsave(MASK_DIR + str(id_) + ".png", np.squeeze(Y_train[n]))

augmentation = iaa.SomeOf((0, 2), [
       iaa.Fliplr(0.5),
       iaa.Flipud(0.5),
       ], random_order=True)

# generates augmented images of batch size 10
b = augmented_image_generator(X_train, Y_train, augmentation, 10)