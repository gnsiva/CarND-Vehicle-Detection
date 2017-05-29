
# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_notcar]: ./output_images/car_not_car.png
[hog]: ./output_images/hog_features.png
[windows]: ./output_images/window_positions.png
[imagedetection]: ./output_images/single_image_detection.png
[debug]: ./output_images/debug_still.jpg
[labels]: ./output_images/labels_map.png
[resultgif]: ./output_images/result.gif
[video1]: ./project_video_result.mp4


## Histogram of Oriented Gradients (HOG)

![Histogram of gradients features][hog]

## Training a classifier

![Classifier run on single image][imagedetection]

## Sliding window search

![alt text][windows]

## Video Implementation

### Filtering by heatmap

![alt text][debug]

### Project submission

![gif of project output][resultgif]

---

# Discussion

