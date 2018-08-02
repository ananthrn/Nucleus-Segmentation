# Author : @kr2741
""" 
Ensembling Mask R-CNN and U-Net (Simple average of MRCNN + Unet)
"""

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

from keras.layers import Input, Conv2D, SeparableConv2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping,ModelCheckpoint

from skimage.morphology import label
from skimage.io import imread

# IOU metric Adapted and modified from https://www.kaggle.com/aglotero/another-iou-metric
# Adapted from @ar3792

def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = label(y_true_in > 0.0)
    y_pred = label(y_pred_in > 0.5)
    
    true_obj = len(np.unique(labels))
    pred_obj = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_obj, pred_obj))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_obj)[0]
    area_pred = np.histogram(y_pred, bins = pred_obj)[0]
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
    precision = []
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
        precison.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(precision)))
    return np.mean(precision)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)

def my_iou_metric(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
    return metric_value

def loadDataAndReturnTrainSet(mask_rcnn_input_folder, u_net_input_folder, gt_mask_folder):
    image_ids = os.listdir(gt_mask_folder)
    X_train = np.zeros((670,512,512,2)) # Input has 2 channels; Channel 1 = UNET, Channel 2 = Mask RCNN
	Y_train = np.zeros((670, 512, 512, 1)) # Output original mask
    for n, image in enumerate(image_ids):
        mask = imread(gt_mask_folder + str(image))[:,:,:1]
        unet_image = imread(u_net_input_folder + str(image))[:,:,:1]
        mrcnn_image = imread(mask_rcnn_input_folder + str(image))[:,:,:1]
        print(np.unique(unet_image))
        X_train[n,:,:,0:1] = unet_image>=100.0
        X_train[n,:,:,1:2] = mrcnn_image <=128
        Y_train[n] = mask >= 128
    return X_train, Y_train

def loadDataAndReturnTestSet(test_mask_rcnn_input_folder, test_u_net_input_folder, test_gt_mask_folder):
    image_ids = os.listdir(gt_mask_folder)
    X_test= np.zeros((65,512,512,2))
	Y_test = np.zeros((65,512, 512,1))
    for n, image in enumerate(image_ids):
        mask = imread(test_gt_mask_folder + str(image))
        print("mask shape:",mask.shape)
        unet_image =imread(test_u_net_input_folder + str(image))[:,:,:1]
        mrcnn_image = imread(test_mask_rcnn_input_folder + str(image))[:,:,:1]
        print("UNET IMAGE:",np.unique(unet_image))
        print("MASK RCNN IMAGE:",np.unique(mrcnn_image))
        print("MASK:",np.unique(mask))
        # setting as two diff channels
        X_test[n,:,:,0:1] = unet_image>=100
        X_test[n,:,:,1:2] = mrcnn_image <=128
        Y_test[n] = (mask.reshape((512,512,1))/65535.0) >= 0.3
    return X_test, Y_test
	    
if __name__ == '__main__':
	print(len(glob.glob('./maskrcnn_final_ensemble_input/*.png')))
	
	# Images not included with submission, available upon request. 
	
	mask_rcnn_input_folder = "maskrcnn_ensemble_input/"
	u_net_input_folder = "../Kernel_Simple/Unet starter/u_net_ensemble_input/"
	gt_mask_folder = "../Kernel_Simple/Unet starter/ensemble_gt_mask/"

	X_train, Y_train = loadDataAndReturnTrainSet(mask_rcnn_input_folder, u_net_input_folder, gt_mask_folder)

	# Sanity Checking the images
	for i in range(40,50):
	    print("Unet:")
	    plt.imshow(X_train[i,:,:,0])
	    plt.show()
	    print("MRCNN:")
	    plt.imshow(X_train[i,:,:,1])
	    plt.show()
	    print("GT:")
	    plt.imshow(np.squeeze(Y_train[i]))
	    plt.show()

	

	earlystopper = EarlyStopping(patience=3, verbose=1,monitor = 'val_loss')
	checkpointer = ModelCheckpoint('ensemble_model_unet_maskrcnn(old_training)_conv_bc_1x1_sigmoid-dsbowl2018-1.h5', verbose=1, save_best_only=True)
	results = ensemble_model.fit(X_train, Y_train, validation_split=0.3, batch_size=4, epochs=10, 
	                    callbacks=[earlystopper, checkpointer])

	
	test_mask_rcnn_input_folder = "../Ensemble-model/maskrcnn2_test_ensemble_input/"
	test_u_net_input_folder = "../Kernel_Simple/Unet starter/unet_test_ensemble_input/"
	test_gt_mask_folder = "../test_data/testMask/"

	X_test, Y_test = loadDataAndReturnTestSet(test_mask_rcnn_input_folder, test_u_net_input_folder, test_gt_mask_folder)
	# Evaluating the ensemble model
    # Taking average across two channels, i.e. Mask R-CNN and U-Net image
    mean_val = np.mean(X_test,axis=3,keepdims=True)
	ensemble_model.evaluate(mean_val,Y_test)