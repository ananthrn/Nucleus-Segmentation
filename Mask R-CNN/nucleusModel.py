# AUTHOR: #kr2741
# Adapted from and modified : https://github.com/matterport/Mask_RCNN/tree/master/samples

import matplotlib.pyplot as plt
import os
import sys
import json
import datetime
import numpy as np
import random

import skimage.io
from imgaug import augmenters as iaa

ROOT_DIR = os.path.abspath("Mask_RCNN/")
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

DATASET_DIR = os.path.abspath("../data/data-science-bowl-2018/stage1_train")
dataset_dir = os.path.abspath("../data/data-science-bowl-2018/")

IMAGE_IDS = os.listdir(DATASET_DIR)
DATASET_LEN = len(IMAGE_IDS)
VAL_IMAGE_IDS = list(random.sample(IMAGE_IDS, int(0.3*DATASET_LEN)))
TRAIN_IDS = list(set(IMAGE_IDS) - set(VAL_IMAGE_IDS))
assert (set(TRAIN_IDS + VAL_IMAGE_IDS) == set(IMAGE_IDS)) == True


class NucleusSegmentationTrainingConfig(Config):
    NAME = "nucleus-segmentation-training"
    IMAGES_PER_GPU = 4
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = len(TRAIN_IDS) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)
    DETECTION_MIN_CONFIDENCE = 0
    BACKBONE = "resnet101"
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000
    RPN_NMS_THRESHOLD = 0.9
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    TRAIN_ROIS_PER_IMAGE = 128
    MAX_GT_INSTANCES = 200
    DETECTION_MAX_INSTANCES = 400

class NucleusSegmentationInferenceConfig(NucleusSegmentationTrainingConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = 0.7

class NucleusSegmentationDataset(utils.Dataset):
    # loads training data 
    def load_training_data(self, dataset_dir):
        self.add_class("nucleus", 1, "nucleus")
        subset_dir = "stage1_train"
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        
        image_ids = next(os.walk(dataset_dir))[1]
        image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        for iid in image_ids:
            self.add_image(
                "nucleus",
                image_id=iid,
                path=os.path.join(dataset_dir, iid, "images/{}.png".format(iid)))
    
    def load_val_data(self, dataset_dir):
        self.add_class("nucleus", 1, "nucleus")
        subset_dir = "stage1_train"
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        image_ids = set(VAL_IMAGE_IDS)

        for iid in image_ids:
            self.add_image(
                "nucleus",
                image_id=iid,
                path=os.path.join(dataset_dir, iid, "images/{}.png".format(iid)))
    
    # loads mask for corresponding image
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    # retrieves path on disk
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

if __name__ == '__main__':
   config = NucleusSegmentationTrainingConfig()
   config.display()
  
   model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir="logs")
   """ 
   PRETRAINED IMAGENET WEIGHTS, available on Keras website
   """ 
   weights_path = model.get_imagenet_weights() 
   """ 
   PRETRAINED COCO WEIGHTS, available at https://github.com/matterport/Mask_RCNN/releases
   """
   # weights_path = "../mask_rcnn_coco.h5" 
   #
   # weights_path = "logs/nucleus-segmentation-training20180426T1505/mask_rcnn_nucleus-segmentation-training_0020.h5"
   
   # model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
   model.load_weights(weights_path, by_name=True)
   
   train_ds = NucleusSegmentationDataset()
   train_ds.load_training_data(dataset_dir)
   train_ds.prepare()

   val_ds = NucleusSegmentationDataset()
   val_ds.load_val_data(dataset_dir)
   val_ds.prepare()

   augmentation = iaa.SomeOf((0, 2), [
       iaa.Fliplr(0.7),
       iaa.Flipud(0.4)
        #iaa.Multiply((0.8, 1.2), per_channel=0.3),
       #iaa.Affine(
      # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
       #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
      #rotate=(-25, 25),
       #shear=(-8, 8)
       ])

   model.train(train_ds, val_ds,
                learning_rate=config.LEARNING_RATE,
                epochs = 20,
                augmentation=augmentation,
                layers='heads')
    
   model.train(train_ds, val_ds,
                learning_rate=config.LEARNING_RATE,
                epochs = 30,
                augmentation=augmentation,
                layers='5+')
   
   model.train(train_ds, val_ds,
                learning_rate=config.LEARNING_RATE,
                epochs = 50,
                augmentation=augmentation,
                layers='all')
