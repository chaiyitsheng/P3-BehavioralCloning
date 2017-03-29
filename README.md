#**Behavioral Cloning** 

This is my submission for CarND-Behavioral Cloning-P3

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

I added an additional goal:

The code/model must work well with a basic machine with the following specs:
Intel Core i5
8GB RAM
Windows 10 64-Bit
& NO GPU (due to budget contraints)

This put a significant handicap on my implementation but I was able to solve it using a novel approach.

1) feature reduction, reducing the size of the images through cropping,thus less features to work on (ie less computation). Remember, I don't have a GPU

2) data pruning, removing data points with zero steering angles reduced straight line bias and again less computation.

The rest I will detail in the section below.



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

This was completed as per project requirements. I tried to keep the code compact by only creating 1) model.py, 2) model.h5 and 3)writeup_report.md. Thus, leaving drive.py unchanged.


####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I initially tried the VGG model and LeNet but found that they did not work well. My model was initially too ambitious,fitting the image, as is (160x320x3). I realised that there was too much noise in data (trees, high tension cables,sky etc).

####2. Attempts to reduce overfitting in the model

Again, one could use dropout layers or multiple data sets to reduce overfitting but I decided to crop the image, removing the top 74, bottom 20, left 80 and right 80. 

In addition, pruning the data points with zero steering angle also reduced overfitting on the entire raw dataset.


####3. Model parameter tuning

I used the adam optimizer built into Keras, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

For details about how I created the training data, see the next section. (3 Creation of the Training Set & Training Process)

###Model Architecture and Training Strategy

####1. Solution Design Approach

For details about how I created the training data, see the section. (3 Creation of the Training Set & Training Process)

1) A major "gotcha" that I have to mention was that I originally used matplotlib.image to read in the image file (matplotlib reads in images in RGB format). So reading in images in this manner didn't work at all. I couldn't figure out why until I did some matrix subtraction of matplotlib.image.imread(file) and cv2.imread(file) and the matrix elements were not zero! I then realised the different formats (cv2 uses BGR vs matplotlib's RGB). Switching to cv2 in model.py, the drive.py file worked!

2) Reading in all the data, resulted in a straight line bias, where the vehicle could keep going straight. I decided to prune all the data points with zero steering angles

3) Also normalized and mean centered the data as a preprocess

4) There was also a lot of noise, so I cropped the image to feed into the Nvidia model. But I made a slight custom adjustment, making the image narrower 160px instead of 200px. This was to reduce the number of features

5)As my final dataset was 5736 items with a training/validation split of 99%/1%. I did not require the use of generator functions.

6) I tested the training /validation split of 99/1%, all the way to 90%/10%. Remember this is not traffic sign recognition, where validation accuracy will have no bearing , if 1) the car cannot drive itself around the track and 2) my computer which did not have a GPU hangs and cannot process it. The model works for all splits between 99/1%, all the way to 90%/10%. 

7) Finally, 5 Epochs did the trick! There is a lot of room for improvement, but this was a first attempt and this the car can get around without using a GPU to build the model. But I think I'll save up for a GPU soon.

Thank you for your kind review. Cheers!

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network as follows:

Layer (type)                     Output Shape          Param #     Connected to                     
======================================================================
lambda_2 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
cropping2d_2 (Cropping2D)        (None, 66, 160, 3)    0           lambda_2[0][0]                   
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 31, 78, 24)    1824        cropping2d_2[0][0]               
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 14, 37, 36)    21636       convolution2d_6[0][0]            
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 5, 17, 48)     43248       convolution2d_7[0][0]            
____________________________________________________________________________________________________
convolution2d_9 (Convolution2D)  (None, 3, 15, 64)     27712       convolution2d_8[0][0]            
____________________________________________________________________________________________________
convolution2d_10 (Convolution2D) (None, 3, 8, 64)      32832       convolution2d_9[0][0]            
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 1536)          0           convolution2d_10[0][0]           
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 1164)          1789068     flatten_2[0][0]                  
____________________________________________________________________________________________________
dense_7 (Dense)                  (None, 100)           116500      dense_6[0][0]                    
____________________________________________________________________________________________________
dense_8 (Dense)                  (None, 50)            5050        dense_7[0][0]                    
____________________________________________________________________________________________________
dense_9 (Dense)                  (None, 10)            510         dense_8[0][0]                    
____________________________________________________________________________________________________
dense_10 (Dense)                 (None, 1)             11          dense_9[0][0]                    
======================================================================

This is based on the NVidia model but I cropped the sides a little smaller (66x160x3) compared to the original (66x200x3) 


####3. Creation of the Training Set & Training Process

The steps taken

1) Recording a) going forward 2 rounds and b) turning around and going in the opposite direction 2 rounds. This is a base dataset. During recording, I drove slowly, deliberately using more key strokes to steer to create more data points. I found that if you drove at full speed, you tend to cruise along certain stretches. During cruising, the recorded steering angle is zero, which leads to overfitting later on.

2) I did not flip the dataset the other way because by going in the opposite direction 2 rounds (the data is essentially flipped).

3) I "augmented" the data by concatenating the center, left and right images as one large set eg

1 data point of centre, left, right, steering angle becomes

centre: steering angle
left: steering angle
right:steering angle

4) Since, I am already driving slowly and delibrately, I did not enhance the recorded steering angle

5) I collected over 6000 datapoints x 3 images for each point, there were over 18000 points in the augmented dataset. This cause my computer to hang when running it. A python generator would have been a good solution.

6) I pruned the data by removing the datapoints with a steering angle of zero resulting in a final dataset of 5736 items. As this was a much more manageable set, I decided to leave out the python generator for the sake of simplicity.   

