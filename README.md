
# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_notcar]: ./output_images/car_notcar.png
[hog]: ./output_images/hog_features.png
[windows]: ./output_images/window_positions.png
[imagedetection]: ./output_images/single_image_detection.png
[debug]: ./output_images/debug_still.jpg
[labels]: ./output_images/labels_map.png
[resultgif]: ./output_images/result.gif
[video1]: ./project_video_result.mp4

## Data

The labelled training images were taken from the GTI database and the KITTI vision benchmark suite. There were 8791 images of cars and 8967 without cars, and each image was 64x64 pixels and had 3 channels. So the dataset is relatively well balanced. Below shows a random selection of the images used, the code for generating the figure is in `figures.ipynb`.

![alt text][car_notcar]


## Sliding window search

In this project we are to cut up the incoming images into several windows and apply a classifier to predict whether the window image is a car or not. The classifier will work on 64x64 images, so the windows will be resized before being fed into it. I used windows of several different sizes and varying amounts of overlap as shown in the figure below. To show the size of a single window relative to the size of the image it has been displayed in the top left corner. 

![alt text][windows]

## Histogram of Oriented Gradients (HOG)

![Histogram of gradients features][hog]

## Training a classifier

When choosing a classifier for this project a key requirement is that it is quick at predicting, as it has to process many windows from each video frame, and then hopefully several frames per second to be useful in the real world. I started out using a `LinearSVC` (linear support vector machine classifier) which was very fast, but not especially accurate (~96%). I then tried a random forest which improved accuracy, and recall which is especially important for correctly identifying false positives, however, it was 10x slower to predict and so couldn't be used. An `SVC` was also used with an `rbf` kernel, but this was much slower than the other two approaches.




![Classifier run on single image][imagedetection]

## Video Implementation

### Filtering by heatmap

![alt text][debug]

### Project submission

![gif of project output][resultgif]

---

# Discussion

