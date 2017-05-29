
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

The `skimage.feature.hog` function was used to extract "histogram of gradient" features from the images. This was done on each channel of the image and is shown below. I used `orient` of 9, as I didn't see much improvement when increasing it. Similarly I tried various values for `pixels_per_cell` and `cell_per_block`, and decided on values of 16 and 2 respectively. I decided not to use the RGB colour space due to the effect of changing light conditions on it. I tried a few other colour spaces and finally decided on HLS.

![Histogram of gradients features][hog]

## Training a classifier

When choosing a classifier for this project a key requirement is that it is quick at predicting, as it has to process many windows from each video frame, and then hopefully several frames per second to be useful in the real world. I started out using a `LinearSVC` (linear support vector machine classifier) which was very fast, but not especially accurate (~96%). I then tried a random forest which improved accuracy, and recall which is especially important for correctly identifying false positives, however, it was 10x slower to predict and so couldn't be used. An `SVC` was also used with an `rbf` kernel, but this was much slower than the other two approaches.

The some of the data is sequential, where several images in a row will be of the same car from different angles/lighting etc. I didn't have time to manually split these, so instead, for the car and non-car images, I took the full list and split off the test set (20%) without shuffling. 

I briefly optimised the `LinearSVC` hyperparameters, I found that lowering the `C` parameter to 0.01 improved performance and did not increase prediction time significantly. The algorithm was tested with both dual (default) and primal optimisation. The later was used as the accuracy produced was better and it predicted slightly faster.

Below is the result of running the classifier on one of the test images. The black squares indicate windows where the classifier predicted that there was a car. In this case I used different windows than described above as the horizon is in a very different position on the y-axis in comparison to the project video.

**See `train_classifier.ipynb`**

![Classifier run on single image][imagedetection]

## Video Implementation

### Filtering by heatmap

False positives were an issue with the classifier, and so the bounding boxes were combined over several frames before they were combined, thresholded and bounding boxed to give the final detection. These steps can be seen in the image below, which is a still shot of the `VehicleDetection.run_debug` output. In the top right we can see the outlines of all the windows where the classifier detected a vehicle for the current frame. The image below is all the detected windows for the previous 50 frames. The histogram is shown in the bottom left, after thresholding, and we can see there are now only two spots which correspond to the two cars. The a bounding box is drawn around these spots to give the final detections, which are shown in the top left.

![alt text][debug]

### Project submission

The full quality project output video is included in this repository as `project_video_result.mp4`. Additionally I have displayed a gif of the video below.

**See `process_video.ipynb`**

![gif of project output][resultgif]


# Discussion

## Speed of prediction

I didn't want to use hard mining as I felt that that was cheating, and training on the test set. So in order to get predictions with no false positives I had to sample a relatively large number of windows and window sizes. In order to compensate for this, I have multithreaded the code so that each window size is calculated in a separate process, and so I can still process 2 frames per second. The multithreading code is in `process_video.ipynb` `VehicleDetection._get_boxes`

**Additional points**
- The HOG gradients were calculated on each image for each window size, given more time I would have changed the code so that these were only calculated once.
- Each individual window was scaled to 64x64, it would have been faster if I scaled the whole image, then cut out windows of 4x64

## Conclusion

I am happy with the performance of the process, it is relatively fast and detects no false positives, which were a big problem in development. If working outside the confines of the project, I would like to attempt the same task but using deep learning. I think the problem is very well suited to convolutional neural networks as we are using image based detection and have quite a large amount of training data.
