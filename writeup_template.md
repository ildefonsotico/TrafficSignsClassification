# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  
1. Submission Files - All files requested was uploaded. The IPython notebook, HTML and a write up report. It will meet the requirements through this criteria. 
2.This criteria make sense be showed by a visualization approach, so it will be shown bellow in order to reach the requirements. 
3. Design and Test a Model Architecture
3.1.Preprocessing - The preprocessing tecnique used was a very simple one. It just normalize each image by the approach taught for the course. Standardize an image using ``(pixel - 128.) / 128.`` for each of the channels (RGB)
3.2. Model Architecture - The architecture used was pretty similar from the LeNet architecture. It was made slighly changes in order to maximize performance and avoid overfitting. The approach used was to add some dropout function using a keep prob of the 0.9. This approach allows increasing from 89% accuracy average to around 96% accuracy. 
3.3. Model Training - The optimiezer tecnique used was AdamOptmizer. It was used by the quiz and it worked pretty well, so it keep in the base architecture. I also did some data augmentation to better generalize the model. I chose 30 Epochs to training the net. The batch size chose was 152. The rate chose was 0.000945. This hyperparameter works better in my approach. I also used the normal size from the LeNet does. 
3.4. Solution Approach - As said above, the acrchitecture used was pretty similar by the LeNet architecture adding just some dropout function in order to increase accuracy and to prevent overfitting. The accuracy reached either it was around 96% or 95.6% to be exact. 
4. Test a Model on New Images - I tried to search on google in order to to get real images from the german signs traffic. At begginer I tried to use some images with bad resolution, then they didn't work well. After that, I tried to put some images like speed limits, and sometimes the net was not able to predict it correctly. It is important enphasizing that I didn't pass them from the pre processing function. By the end, and After I preprocessing the images found by the google, the prediction work pretty well. I found in some dataset 80% of accuracy. 
 3.2.Model Architecture - 
---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ildefonsotico/TrafficSignsClassification/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The data shown bellow are chose randomically.It was extract by the dataset used from the german laboratory. 

![trafficsign_dataexploration](https://user-images.githubusercontent.com/19958282/40462028-84c00470-5ee4-11e8-8e8e-ae252ad946e6.png)

There are a non-homogenous samples for each kind of image. It can cause issue and misunderstanding about how the network can handle with the dataset. Bellow will be shown how each class of images are distributed in the dataset. It will be divided by Training Set, Validation Set and Testing Set.   

![traffic_ytrain_samples](https://user-images.githubusercontent.com/19958282/40462316-cf7491b0-5ee5-11e8-90a1-684b2a293074.png)

![traffic_y_validation](https://user-images.githubusercontent.com/19958282/40462317-d09e3c26-5ee5-11e8-88d6-9b78d82da90e.png)

![traffic_y_validation_samples](https://user-images.githubusercontent.com/19958282/40462318-d1b65b16-5ee5-11e8-93af-ee84f27fa801.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


At beginner, I normalized the image data because using a simple approach. It just standartize each image.
![normalization](https://user-images.githubusercontent.com/19958282/40462642-38166df0-5ee7-11e8-97f3-a0fc248b2187.png)

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model of the architecture used can be found by the 10 cell in the IPython.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Dropout with Max polling 5x5     	| 2x2 stride, same padding, outputs 14x14x6 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| Dropout with RELU    	| Activation 	|
| Dropout with Max polling 5x5     	| 2x2 stride, same padding, outputs 5x5x16 	|
| Fully connected		| input 400, output 120        									|
| Dropout with RELU    	| Activation 	|
| Fully connected		| input 120, output 84        									|
| Dropout with RELU    	| Activation 	|
| Fully connected		| input 84, output 43        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The traning architeture can be found by the 10th section of the IPython. 

To train the model, I used an variation of the LeNet architecture. This architecture used a convolution following with a Dropout and maxpooling with a stride (2x2). It was used dropout in order to allow better avoid overfitting. The next step was used a new Convolution with a stride (1x1) following by the dropout with a RELU activation funtion. Next step was used again a dropout with maxpooling as shown above, following by 3 fully connected layers intercalated by dropout between them. The epochs used was 30 with a batch_size of 128. The rate chose was 0.00995. It works well and provided between 94,5 to 95.8% of the accuracy. I guess the data augmentation a better disturb inserted on the samples could help to improve the accuracy. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.950 
* test set accuracy of 0.937

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I began using the LeNet architecture, but it was not able to provide more than 89% of accuracy.
* What were some problems with the initial architecture?
I guess the main problem was the overfitting that architectura was getting. 
* How was the architecture adjusted and why was it adjusted? 
As I said before, I guess the mainly problem was the overfitting, so I added some dropout funtion combined by maxpooling or relu activation funtion. It was done to try increasing accuracy and avoid overfitting. 
* Which parameters were tuned? How were they adjusted and why?
The number of epochs was incresed for 20. The learning rate was chose to be 0.00995. It was added dropout function where was used a Keep_prob by 0.9.
* What are some of the important design choices and why were they chosen? 
The convolution was used same LeNet because it is able to classifie better image objects. Dropout was used to prevent overfiting. MaxPooling was used to improve the accuracy and classification. 

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![0_sign](https://user-images.githubusercontent.com/19958282/40570833-659423b0-6065-11e8-90b4-361ba36008b4.jpg) ![14_sign](https://user-images.githubusercontent.com/19958282/40570836-67b361ce-6065-11e8-8e42-3f1f1f6677a5.jpg) ![17_sign](https://user-images.githubusercontent.com/19958282/40570839-6b15a7dc-6065-11e8-8a0f-1e208d71d0cf.jpg) ![25_sign](https://user-images.githubusercontent.com/19958282/40570840-6d132d16-6065-11e8-8d22-a075538edc2f.jpg)
![33_sign](https://user-images.githubusercontent.com/19958282/40570841-6f93f2f0-6065-11e8-98b2-f5d52d3520bd.jpg)

The first image might be difficult to classify because it is same than others. I guess the net could identify the relevants objects on the image, but it was not able to identified correcly which speed limit there is inside. 
The other images the Net classified well. 
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 20Km limit      		| No vehicles   									| 
| Stop     			| Stop 										|
| No entry					| No entry											|
| Road work	      		| Road work					 				|
| Turn right ahead			| Turn right ahead     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 36th cell of the Ipython notebook.

![softmax1](https://user-images.githubusercontent.com/19958282/40571136-d7904210-6069-11e8-8c7f-412ab6fa2e1e.png)
![softmax2](https://user-images.githubusercontent.com/19958282/40571137-d855cdaa-6069-11e8-835e-d7c686273a4e.png)
