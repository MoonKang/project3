**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./steering_dist.png "Steering distribution"
[image2]: ./steering_hist.png "Steering Histogram"
[image3]: ./img.png "Image1"
[image4]: ./img_adj.png "Image2"
[image5]: ./image_adj.png "Image3"
[image6]: ./steering_adj.png "Steering New Histogram"

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

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of 4 convolution neural network layers with 3x3, 2x2 filter sizes and depths of 32, 64, 128, and 128. (model.py lines 243-268) 

The model includes RELU layers to introduce nonlinearity after each cnn layer, and the data is normalized in the model using a Keras lambda layer (code line 250). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 259,261,263). After trying a few variations for dropout rate, I have found 0.5 to work well in mitigating bias toward going straight. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 71). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer and have set parameters to be following: (lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0). 

####4. Appropriate training data

Initially I have used Udacity driving images for training. From browsing through Udacity data I found that there weren't not sufficient data for  the curve by dirt road after bridge. I have created my own data to supplement and also added more recovery data. 

I found that data fluctuation is quite consistent over time frame and \\
![alt text][image1]

that  data is highly concentrated near 0 steering angle. \\

![alt text][image2]

I have used a combination of center, right, and left images and have adjusted +- 0.22 for left and right images. 

![alt text][image3]

Then I have 1) shifted, adjusted brightness, shadowed, and flipped images. 

![alt text][image4]

Then I have resized images to be 64x64 for training.

![alt text][image5]


###Model Architecture and Training Strategy

####1. Solution Design Approach


My first step was to use a convolution neural network architecture similar to the one provided by Nvidia. I was able to get good training and valid loss but I was not able to complete a lap on track with the trained model. Then I added max pool layers between cnn so that the model can be more focused in training. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model to include dropouts. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.  To improve the driving behavior in these cases, I have added more data for curves at dirt road after bridge. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model2) consists of 4 convolution neural network layers with 3x3, 2x2 filter sizes and depths of 32, 64, 128, and 128. (model.py lines 243-268). The data is normalized in the model using a Keras lambda layer (code line 250). Between cnn layers I have added RELU for nonlinearity and maxpooling to focus on neurons that mattered the most. 

####3. Creation of the Training Set & Training Process

I have combined Udacity data and my own to supplement it. As described at 'Appropriate training data' I have added variations to prevent for brightness, shadow, and positional bias. From data histogram I have noticed that large portion of data has close to 0 steering angle. As a result, I have noticed that the car had tendency of going straight despite of curvatures in lane. In order to avoid the problem, I have manually lowered the percentage of data that has 0 steering degree and data histogram has been changed: 

![alt text][image6]

I finally randomly shuffled the data set and put 10% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 14 as evidenced by trials with epoch of 10 and 15.