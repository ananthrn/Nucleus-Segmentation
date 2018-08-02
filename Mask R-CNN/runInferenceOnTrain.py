# AUTHOR: #kr2741
# Adapted from and modified  : https://github.com/matterport/Mask_RCNN/tree/master/samples

""" 
Running inference on train set to generate input for ensemble model
""" 

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
ROOT_DIR = os.path.abspath("Mask_RCNN/")

sys.path.append(ROOT_DIR)  # To find local version of the library
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
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)
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

dataset = nucleus_updated.NucleusDataset()
dataset.load_nucleus(DATASET_DIR, "stage1_train")
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
    model_dir="logs/",
    config=config)

weights_path = "logs/nucleus-segmentation-training20180426T0614/mask_rcnn_nucleus-segmentation-training_0022.h5"

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
ctr = 0
print(len(dataset.image_ids))

for image_id in dataset.image_ids:
    print("Image id: ", image_id)
    filename = "maskrcnn_ensemble_input/" + str(dataset.image_reference(image_id)) + ".png"
    #if os.path.isfile(filename):
     #   continue
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    mrcnn = model.run_graph([image], [
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
            ("masks", model.keras_model.get_layer("mrcnn_mask").output),
            ])

    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    det_count = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:det_count]

    print("{} detections: {}".format(det_count, np.array(dataset.class_names)[det_class_ids]))

    det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], (512,512))
    det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c] 
                                  for i, c in enumerate(det_class_ids)])
    det_masks = np.array([utils.unmold_mask(m, det_boxes[i], (512,512,3))
                          for i, m in enumerate(det_mask_specific)])
    log("det_mask_specific", det_mask_specific)
    log("det_masks", det_masks)
    print(np.unique(det_masks))
    a = np.sum(det_masks, axis=0) > 0
    # create directory using terminal if it doesn't already exist
    plt.imsave(fname = "maskrcnn_final_ensemble_input/" + str(dataset.image_reference(image_id)+".png"), arr = a, cmap="Blues")
