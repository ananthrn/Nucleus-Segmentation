# Nucleus-Segmentation
Architectures and baseline models for the Nucleus Segmentation Challenge as part of the 2018 Data Science Bowl hosted by Kaggle. 

# Directory Structure 
* Weighted U-Net/ - contains all scripts required to run weighted U-net model including preprocessing. 

* U-Net/ - contains all scripts required to run diff versions of u-net

* Mask R-CNN/ - contains all scripts required to run mask r-cnn

* Ensemble/ - contains all scripts for the ensemble models

# References used
* https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277?scriptVersionId=2164855  - used to construct baseline U-Net model, and is further modified to construct other versions of U-Net and is then used in ensembling U-Nets and U-Net and Mask R-CNN. 

* https://www.kaggle.com/aglotero/another-iou-metric - used to build evaluation metric to establish standard way of evaluating all our models. 

* https://github.com/matterport/Mask_RCNN/tree/master/samples - used as reference to construct MRCNN model to establish a baseline and then use in ensembling Mask R-CNN and U-Net

Please feel free to contact me at ananth360@gmail.com for any images/weights required to run the model. 
