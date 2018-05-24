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

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

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

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


