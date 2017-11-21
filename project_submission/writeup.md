[//]: # (Image References)
[model_architecture]:./images/Network_architecture.png

# Project: Follow Me (Deep Learning)

## 1. Program Structure
   My first step in this project was to modify the organization of the project code so I could more easily test a variety of settings on AWS without having to start each run manually. I chose to break the primary functions of the code into a few files:
   * [AWS_training_dispatcher.py](https://github.com/croomjm/RoboND-DeepLearning-Project/blob/master/code/AWS_training_dispatcher.py): This is the high level training run dispatcher. It sets the hyper parameters for each run, generates the weights file name from the parameters, and calls the `train_model()` function from AWS_training_utils.py.
   * [AWS_training_utils.py](https://github.com/croomjm/RoboND-DeepLearning-Project/blob/master/code/AWS_training_utils.py): This is where the magic happens. All of the training layers are defined here as well as a high level function that sets up, initializes, and saves the results from the network training.
   * [AWS_check_all_models.py](https://github.com/croomjm/RoboND-DeepLearning-Project/blob/master/code/AWS_check_all_models.py): Typing is hard. That's why I chose to make a processing function to test the performance of all weights files in the weights directory and compile the results into a simple-to-read github markdown table sorted by overall network score (definition of score below).
   * [download_images.sh](https://github.com/croomjm/RoboND-DeepLearning-Project/blob/master/code/utils/download_images.sh): I chose to gather supplemental training/validation data in addition to the baseline data provided as part of the project. I stored this data in an AWS S3 bucket (how is this free?!) for easy access and download. Unfortunately, that meant that I would have to type 3 long wget commands and move all the images into appropriate folders. Uacceptable. That's why I wrote this slick bash script to pull all of the images (including the Udacity-provided images) from their respective S3 buckets, unzip them, and recombine them into the right subfolders. I also included options to either download or not download the supplemental training or validation data to make it easy to compare the performance with various datasets.

  The task at hand is fairly simple: Given a camera view of a scene, indentify which pixels within in the scene can be classified as the "hero" (target the drone would like to follow). As a byproduct, we're also interested in identifying pixels that below to the background or represent people other than the hero. Since we're able to gather simulated images from Unity, it's easy to generate training data as the game engine knows a priori which pixels in the scene can be classified as background, hero, or other person. As a result, the most difficult part of the task is to use this tagged training data in combination with an appropriate neural network architecture to quickly segment the scene into pixels of each class.

## 2. Network Architecture

  This type of task lends itself to an FCN (fully convolutional network) for one primary reason: An FCN preserves spatial information. We need to identify where in an image an object exists, not just whether it's in the image at all. This is accomplished by using only convolutional layers rather than fully connected layers, which collapse the 4D tensor into lower dimensional space, destoying spatial information in the process. Although this isn't relevant for this problem, FCNs also have the benefit of handling arbitrarily shaped input images, which is useful for general image segmentation problems (e.g. Google image search classification of objects within crawled images of arbitrary size).
  
  The general FCN is constructed with the following elements:
   1. N encoder layers, each containing a separable convolution layer with relu activation and output batch normalization
   2. 1 1x1 convolution layer with batch normalization
   3. N decoder layers, each containing a separable convultion layer with relu activation and output batch normalization (and possibly a skip connection from prior layer(s) concatenated using bilinear upsampling)
   4. Final convolution layer with softmax activation
   
  Explanation of layer types:
   1. Separable Convolution Layer: The separable convolution layer works by traversing a convolution kernal across each layer of the input , then combining the results of the output of each layer using N separate 1x1 convolutions for each layer and adding the results together to get the final result (where N is the number of output layers). The primary effect is to reduce the number of parameters compared to traversing each layer with N separate kernels per input layer, which in turn increases training speed.
   2. Batch Normalization: Rather than simply normalizing all inputs to the neural network, normalizing the input to every layer in the network with respect to the images in each batch reduces the covariance of each batch of images within each layer. This regularization of the data allows the weights to converge more quickly and increases training stability. 
   3. Skip Connnections: Skip connections help to improve the output resolution by feeding the output from previous layers (before fully collapsed using the 1x1 convolution) to subsequent layers, essentially allowing some higher level spatial information to pass further toward the neural network output without downsampling.
   4. Bilinear Upsampling: In this case, bilinear upsampling was used to combine the output of earlier layers with the input to subsequent layers. In order to do so, the output from earlier layers needs to be increased in size to match the input to later layers. In this project, all bilinear upsampling increased the size of the image by a factor of 2 using a weighted average of nearby pixels (though this isn't a requirement in general).
   
   To keep the number of features from exploding too much, I chose to use 3 encoder/decoder layers with 2 skip connections (see below). This showed relatively good performance in the labs performed as part of earlier lessons and trained fairly quickly on the AWS EC2 p2.xlarge instance. It's possible other, more complex architectures may have resulted in better performance, but this seemed like a good compromise given limited time.
   
![Model Architecture][Model_architecture]

## 3. Tuning Hyperparameters
 The model has the following hyperparameters:
 1. Batch Size: This defines the number of training images that are propagated through the model during a single epoch step.
 2. Number of Epochs: This defines how many full training iterations are used to train the model weights. During each epoch, training data is propagated through the network. At the start of each epoch, the weights are initialized with the output of the previous epoch.
 3. Learning Rate: This defines the step size used when modifying each weight using the gradient descent optimizer. Note that in some of the optimizers (e.g. Nadam), this rate changes over time according to optimizer settings.
 4. Steps per Epoch: This defines the number of image batches that are used in each training epoch to train the model. In order to use all of the available training images, this number should approximately equal (number of training images)/(batch size).
 5. Validation Steps: This defines the number of image batches that are used in each validation epoch to train the model. In order to use all of the available training images, this number should approximately equal (number of training images)/(batch size).
 6. Optimizer: There are a number of optimizers available to perform gradient descent optimization on the model weights. This hyperparameter controls which on is use to train the model.
 7. Number of Workers: This controls the number of threads that can be created by tensorflow/keras to train the model. This isn't really a hyper parameter for the training itself since it shouldn't affect the results, but it does affect how quickly the training is completed. 
 
 Although the default optimizer, Adam, worked fine, I looked into alternatives and found that Nadam gave me slightly better results. Nadam is similar to Adam, except that it uses Nesterov momentum. I found [this article](http://ruder.io/optimizing-gradient-descent/) helpful in developing my understanding of the differences between the available optimization algorithms and why Nadam might be a good choice. Both Nadam and Adam move in the direction of the gradient at the current position, but the direction of descent is augmented by taking into account the previous descent directions. This serves to apply a sort of time constant to the descent direction such that the algorithm continues along a smoother path even if local gradient changes are fairly noisy. Nesterov momentum, however, improves upon this method by not only using previous descent directions to add momentum to the optimizer but also using an estimate of the gradient at the future position to modulate how far along this descent direction it moves. In essence, if it sees that it will be approaching a hill ahead, it slows itself down to avoid ascending up the hill. For similar parameters, I achieved better results with Nadam than Adam (parameters set per recommendations from [Keras documentation](https://keras.io/optimizers/)).
 
 I ran a series of tests to determine which hyperparameters would be most successful. Note that these tests were performed with different datasets. All runs labeled 'default' were run with the Udacity-provided dataset. After seeing the performance of these runs, I chose to gather more training/validation images of a few scenarios: hero in a dense crowd, far away from the hero, and no hero with and without a dense crowd. I saw marginal improvements with the expanded datasets, but not nearly as much as I expected.
 
 In any case, I successfully achieved the minimum passing score of 0.40 for a number of configurations, both with and without supplemental training/validation data.

 | Run | Learning Rate | Batch Size | Epochs | Steps per Epoch | Validation Steps | Optimizer | IOU | Score | Dataset |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.005 | 64 | 75 | 103 | 110 | Nadam | 0.569 | 0.425 | Supplemental Training & Validation Data |
| 2 | 0.005 | 64 | 125 | 103 | 110 | Nadam | 0.547 | 0.415 | Supplemental Training & Validation Data |
| 3 | 0.005 | 256 | 12 | 500 | 50 | Nadam | 0.562 | 0.411 | Supplemental Training Data |
| 4 | 0.005 | 64 | 15 | 400 | 50 | Nadam | 0.535 | 0.405 | Default |
| 5 | 0.005 | 64 | 12 | 400 | 50 | Nadam | 0.541 | 0.398 | Supplemental Training Data |
| 6 | 0.005 | 64 | 40 | 103 | 110 | Nadam | 0.540 | 0.393 | Supplemental Training & Validation Data |
| 7 | 0.005 | 128 | 15 | 400 | 50 | Adam | 0.537 | 0.392 | Default |
| 8 | 0.01 | 256 | 15 | 500 | 50 | Nadam | 0.529 | 0.390 | Default |
| 9 | 0.01 | 64 | 20 | 400 | 50 | Nadam | 0.562 | 0.390 | Supplemental Training & Validation Data |
| 10 | 0.005 | 64 | 15 | 400 | 50 | Adam | 0.539 | 0.389 | Default |
| 11 | 0.002 | 64 | 25 | 400 | 50 | Nadam | 0.535 | 0.387 | Supplemental Training & Validation Data |
| 12 | 0.002 | 64 | 15 | 500 | 50 | Adam | 0.535 | 0.386 | Default |
| 13 | 0.005 | 64 | 20 | 400 | 50 | Nadam | 0.526 | 0.381 | Supplemental Training & Validation Data |
| 14 | 0.01 | 256 | 15 | 500 | 50 | Adam | 0.518 | 0.377 | Default |
| 15 | 0.005 | 128 | 12 | 400 | 50 | Nadam | 0.520 | 0.374 | Supplemental Training Data |
| 16 | 0.01 | 128 | 15 | 500 | 50 | Adam | 0.506 | 0.370 | Default |
| 17 | 0.01 | 64 | 15 | 500 | 50 | Adam | 0.505 | 0.363 | Default |
| 18 | 0.002 | 128 | 15 | 500 | 50 | Adam | 0.501 | 0.361 | Default |


| Run | Overall<br>Score | Following Target<br>Percent False Negatives | No Target<br>Percent False Positives | Far from Target<br>Percent False Positives | Far from Target<br>Percent False Negatives | Dataset |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.425 | 0.0% | 28.413% | 0.929% | 47.678% | Supplemental Training & Validation Data |
| 2 | 0.415 | 0.184% | 19.926% | 0.619% | 49.536% |Supplemental Training & Validation Data |
| 3 | 0.411 | 0.0% | 27.306% | 0.619% | 52.632% | Supplemental Training Data |
| 4 | 0.405 | 0.0% | 16.236% | 0.31% | 52.941% | Default |
| 5 | 0.398 | 0.0% | 26.199% | 0.929% | 51.703% | Supplemental Training Data |
| 6 | 0.393 | 0.0% | 28.782% | 0.619% | 52.941% | Supplemental Training & Validation Data |
| 7 | 0.392 | 0.0% | 18.819% | 0.31% | 58.204% | Default |
| 8 | 0.39 | 0.0% | 12.915% | 0.619% | 60.062% | Default |
| 9 | 0.39 | 0.0% | 48.708% | 1.238% | 50.464% | Supplemental Training & Validation Data |
| 10 | 0.389 | 0.0% | 23.985% | 0.929% | 57.276% | Default |
| 11 | 0.387 | 0.0% | 19.188% | 0.929% | 59.752% | Supplemental Training & Validation Data |
| 12 | 0.386 | 0.0% | 28.413% | 0.31% | 55.108% | Default |
| 13 | 0.381 | 0.0% | 22.509% | 0.31% | 57.895% | Supplemental Training & Validation Data |
| 14 | 0.377 | 0.0% | 16.974% | 0.619% | 60.372% | Default |
| 15 | 0.374 | 0.0% | 19.188% | 0.31% | 60.991% | Supplemental Training Data |
| 16 | 0.37 | 0.0% | 10.332% | 0.31% | 63.158% | Default |
| 17 | 0.363 | 0.184% | 16.236% | 0.31% | 62.539% | Default |
| 18 | 0.361 | 0.0% | 13.653% | 0.31% | 64.396% | Default |
 
 Note: I ran a number of these tests before I realized that I was assigning the batch size and training/validation steps per epoch incorrectly. Ideally, the number of training and validation runs should be approximately equal to the number of images in each dataset divided by the size of each batch so that all images are used about once each iteration. With my arbitrarily assigned batch sizes and number of steps, I was running each image more than once during each training epoch. I assume that this behaves similarly to a model for with an equivalent batch size but larger number of epochs, but I'm not sure how or if the data generator subdivides the data set into batches (at random or ensuring each is taken from a shuffled stack before reusing) or if the optimizer alters the learning rate between epochs and not within an epoch.

## 4. Final Model Performance
 Here's a video of my [best performing model](https://github.com/croomjm/RoboND-DeepLearning-Project/blob/master/data/weights/weights_005_rate_64_batch_75_epochs_103_epoch_steps_110_valid_steps_Nadam_opt_20171120-203242) in action! Even with the seemingly mediocre performance, the model still does a decent job acquiring the hero, and it does a great job following the hero once she's acquired. This behavior aligns well with the data gathered during the training and validation phase since the percentage of false negatives while following closely behind the hero is particularly low.

 [![Successfully Following the Hero!](https://img.youtube.com/vi/Nr_QVikSQto/0.jpg)](https://www.youtube.com/watch?v=Nr_QVikSQto)

## 5. Conclusions
 I can think of a few strategies that might help me improve my results further:
 1. Use a proven CNN architecture (e.g. VGG16) as the basis for my FCN.
 2. Add more layers to the network and train for more epochs.
 3. Gather more data to improve my model's performance, especially of the hero from far away and of other people from far away.
 4. Add dropout to my model and train for many more epochs.
 
 The model as is would probably be able to accommodate segmenting out dogs, cats, cars, etc., though additional training would be required to do so. In general, the features represented by the weights as currently trained would not translate well to tracking different objects. In order to track a larger number of object classes, it might also be necessary to increase the depth and complexity of the network to capture the full breadth of features required to identify the classes.

 I'm generally amazed not only by how powerful neural networks are but also how easy it's become to implement them thanks to a number of open souce libraries. There's still plenty to learn, but I'm excited to experiment!
