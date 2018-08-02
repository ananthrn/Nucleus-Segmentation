# AUTHOR: #kr2741
# Adapted from and modified  : https://github.com/matterport/Mask_RCNN/tree/master/samples

""" 
Running inference on test set to generate test input for ensemble model
""" 

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import skimage.io
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("Mask_RCNN/") # Point to where the Mask RCNN library is in your local system
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config

DATASET_DIR = "/home/kr2741/data/data-science-bowl-2018/"

import nucleusModel
stage_1_train  = os.path.abspath("../data/data-science-bowl-2018/stage1_train")

IMAGE_IDS = os.listdir(stage_1_train)
DATASET_LEN = len(IMAGE_IDS)
VAL_IMAGE_IDS = list(random.sample(IMAGE_IDS, int(0.3*DATASET_LEN)))
TRAIN_IDS = list(set(IMAGE_IDS) - set(VAL_IMAGE_IDS))
assert (set(TRAIN_IDS + VAL_IMAGE_IDS) == set(IMAGE_IDS)) == True

class NucleusSegmentationInferenceConfig(Config):
    IMAGES_PER_GPU = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = 0.7
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = len(TRAIN_IDS) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, 65 // IMAGES_PER_GPU)
    DETECTION_MIN_CONFIDENCE = 0 
    BACKBONE = "resnet50"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    TRAIN_ROIS_PER_IMAGE = 128
    MAX_GT_INSTANCES = 200
    DETECTION_MAX_INSTANCES = 400
    NAME = "nucleus-segmentation-training"

config = NucleusSegmentationInferenceConfig()
config.display()
DEVICE = "/gpu:0"
TEST_MODE = "inference"

class TestDataset(utils.Dataset):
    # kr2741
    def load_X(self, dataset_dir):
        self.add_class("nucleus", 1, "nucleus")
        dataset_dir = "../data/data-science-bowl-2018/stage1_test"
        image_ids = next(os.walk(dataset_dir))[1]
        print("Total", len(image_ids))
        for iid in image_ids:
            self.add_image("nucleus", image_id=iid, path=os.path.join(dataset_dir, iid, "images/{}.png".format(iid)))

    def load_mask(self, image_id):
        dd = "test_data/testMask/"
        info = self.image_info[image_id]
        print(image_id)
        print(info["id"])
        mask = []
        m = skimage.io.imread(os.path.join(dd, info["id"] + ".png")).astype(np.bool)
        mask.append(m)
        mask = np.stack(mask, axis=-1)
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


dataset = TestDataset()
dataset.load_X("test_data/")
dataset.prepare()

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
    model_dir="logs/",
    config=config)

# Weights not included with submission due to size, available upon request. 
weights_path = "logs/nucleus-segmentation-training20180426T0614/mask_rcnn_nucleus-segmentation-training_0022.h5" 

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

ctr = 0

import os.path
for iid in dataset.image_ids:
    print("Image id: ", iid)
    filename = "maskrcnn_final_test_ensemble_input/" + str(dataset.image_reference(iid)) + ".png"
    if os.path.isfile(filename):
        continue

    image, imageMeta, groundTruthClassId, groundTruthBoundingBox, groundTruthMask = modellib.load_image_gt(dataset, config, iid, use_mini_mask=False)
    maskrcnnOutput = model.run_graph([image], [
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
            ("masks", model.keras_model.get_layer("mrcnn_mask").output),
            ])

    detectedClasses = maskRCnnOutput['detections'][0, :, 4].astype(np.int32)
    numberOfDetections = np.where(det_class_ids == 0)[0][0]
    detectedClasses = detectedClasses[:numberOfDetections]

    print("{} detections: {}".format(numberOfDetections, np.array(dataset.class_names)[detectedClasses]))

    detectedBoundingBoxes = utils.denorm_boxes(maskrcnnOutput["detections"][0, :, :4], (512,512))
    detectedmasks = np.array([maskRCnnOutput["masks"][0, i, :, :, c] 
                                  for i, c in enumerate(detectedClasses)])
    
    # detectedMasks reshaped to (512,512,3) to use as input for ensemble model 
    detectedMasks = np.array([utils.unmold_mask(m, detectedBoundingBoxes[i], (512,512,3))
                          for i, m in enumerate(detectedmasks)])
    print(np.unique(detectedMasks))
    a = np.sum(detectedMasks, axis=0) > 0
    # create directory using terminal if it doesn't already exist
    plt.imsave(fname = "maskrcnn_final_test_ensemble_input/" + str(dataset.image_reference(iid)+".png"), arr = a, cmap="Blues")
