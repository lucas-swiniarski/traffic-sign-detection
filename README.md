# traffic-sign-detection

This project is an implementation of a Kaggle competition. 98.4 % accuracy on the classification task.

# Goal

Classify correctly traffic signs in Torch.

![Image not found](images/traffic-sign-1.png "Data example")

# Data augmentation visualization :

Given an image with a traffic sign, center the traffic sign :

![Image not found](images/traffic-sign-1.png "Before Centering")
![Image not found](images/traffic-sign-1-centered.png "After Centering")

Then apply various image transformations described after.

![Image not found](images/traffic-sign-2.png "After Centering")

# Implementation

* data: folder containing the train/test data tensors.
* images: images for readme.
* models: Different convnet models implemented.
* notebooks: Visualizing the data, preprocessing transformations and data augmentation.
* main.lua: load data, model, and train/evaluate the model.
* preprocess.lua: normalize the train/test set. ZCA (or de-correlation) not implemented yet.
* utils.lua: Data transformations utils, examples are :
  * Center the traffic sign in an image: Given an image and the coordinate of the traffic sign in it, return the image centered on the traffic sign.
  * change image contrast
  * rotation.
  * translation.
  * saturation.
  * brightness.
