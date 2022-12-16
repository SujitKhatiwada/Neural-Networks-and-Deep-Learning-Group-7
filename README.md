# Weed detection using MaskRCNN
This repository contains the project details for Group 7, Term 3, Big Data Analytics at Lambton College.

# Introduction to CNN:
ConvNets are the superheroes that took working with images in deep learning to the next level. With ConvNets, the input is an image ,or more specifically, a 3D Matrix. In simple word what CNN does is, it extract the feature of image and convert it into lower dimension without losing its characteristics

# Transfer Learning:
Transfer learning makes use of the knowledge gained while solving one problem and applying it to a different but related problem.
For example, knowledge gained while learning to recognize cars can be used to some extent to recognize trucks.

# Background of MaskRCNN

- RCNN
- FastRCNN
- FasterRCNN
- Network Backbone

- Region Proposal Network
- RoI(Region of Interest)  Pooling
- RoI(Region of Interest) Align

- Network Head
- Classification and Detection
- Segmentation


### Dataset Building
The main dataset used by MaskRCNN is MS COCO dataset which has 80 classes and 115 thousand training images.Evaluation metrics for bounding boxes and segmentation mask is based on Intersection over Union.We use the pretrained weight that model has learned on this datasets and used to train with our own datasets.

Our dataset consists of 202 images in which 183 images were used for training and 19 images for validation.
Since we are going to train an instance segmentation model that should have pixel level accuracy it's important to annotate the 
images properly this has used [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/) tool for this purpose. We took this
data from [GITHUB](https://github.com/AjinJayan/weed_detection/blob/master/dataset_updated.zip).

### Model Weights and Tensorboard Logs
[Pre-trained weight](https://drive.google.com/file/d/11XssW0dkMGfxsFWM-zp_DxICXsLqnGtf/view?usp=sharing) --to be updated

weed_detection_maskrcnn.ipynb    ---- This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline.

weed_training.py ---- This file is the sub-class of the config.py file in the mrcnn module.You can edit this file to tweak the hyperparameters like leaning rate,no.of iteration etc.You can also add custom callbacks from tf.keras for custom logging into tensorboard

### Using Tensorboard for visulatization and model graphs

Once the model completes it's learning.You will be able to download the log file(tf.events) or you can use tensorboard directly in iPython notebook using magic commands

For visualising in system.You must have tensorflow installed and place the log file in logs directory.Open command propmt and type
```
tensorboard --logdir=logs
```


This project was made as part of final project of AML-3014 Neural Network and Deep Learning,  by [Sujit Khatiwada](mailto:sujitkhatiwada07@gmail.com?subject=[GitHub]%20Weed%20Detection). 

### Citation
+ [Weed Detection GitHub](https://github.com/AjinJayan/weed_detection) 
+ [MaskRCNN Paper](https://arxiv.org/pdf/1703.06870.pdf)
+ [Mask_RCNN GitHub](https://github.com/matterport/Mask_RCNN)
+ [FasterRCNN Paper](https://arxiv.org/pdf/1504.08083.pdf)
