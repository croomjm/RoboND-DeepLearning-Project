[//]: # (Image References)
[normalized_confusion_matrix]: ./images/Normalized_confusion_matrix.png
[raw_confusion_matrix]: ./images/Raw_confusion_matrix.png
[shadow_puppets]: ./images/shadow_puppets.png
[test_world_1_result]: ./images/test_world_1.png
[test_world_2_result]: ./images/test_world_2.png
[test_world_3_result]: ./images/test_world_3.png

# Project: Follow Me (Deep Learning)

## 1. Program Structure
   My first step in this project was to modify the organization of the project code so I could more easily test a variety of settings on AWS without having to start each run manually. I chose to break the primary functions of the code into a few files:
     * [AWS_training_dispatcher.py](https://github.com/croomjm/RoboND-DeepLearning-Project/blob/master/code/AWS_training_dispatcher.py): This is the high level training run dispatcher. It sets the hyper parameters for each run, generates the weights file name from the parameters, and calls the `train_model()` function from AWS_training_utils.py.
     * [AWS_training_utils.py](https://github.com/croomjm/RoboND-DeepLearning-Project/blob/master/code/AWS_training_utils.py): This is where the magic happens. All of the training layers are defined here as well as a high level function that sets up, initializes, and saves the results from the network training.
     * [AWS_check_all_models.py](https://github.com/croomjm/RoboND-DeepLearning-Project/blob/master/code/AWS_check_all_models.py): Typing is hard. That's why I chose to make a processing function to test the performance of all weights files in the weights directory and compile the results into a simple-to-read github markdown table sorted by overall network score (definition of score below).
     * [download_images.sh](https://github.com/croomjm/RoboND-DeepLearning-Project/blob/master/code/utils/download_images.sh): I chose to gather supplemental training/validation data in addition to the baseline data provided as part of the project. I stored this data in an AWS S3 bucket (how is this free?!) for easy access and download. Unfortunately, that meant that I would have to type 3 long wget commands and move all the images into appropriate folders. Uacceptable. That's why I wrote this slick bash script to pull all of the images (including the Udacity-provided images) from their respective S3 buckets, unzip them, and recombine them into the right subfolders. I also included options to either download or not download the supplemental training or validation data to make it easy to compare the performance with various datasets.

  The task at hand is fairly simple: Given a camera view of a scene, indentify which pixels within in the scene can be classified as the "hero" (target the drone would like to follow). As a byproduct, we're also interested in identifying pixels that below to the background or represent people other than the hero. Since we're able to gather simulated images from Unity, it's easy to generate training data since the game engine knows a priori which pixels in the scene can be classified as background, hero, or other person. As a result, the most difficult part of the task is to use this tagged training data in combination with an appropriate neural network architecture to quickly segment the scene into pixels of each class.

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
   
   

## 3. Tuning Hyperparameters
## 4. Final Model Performance
 ```python
 def main():
    model_file = '../training/model_100_orientations_sigmoid_YCbCr_16_bin.sav'

    # ROS node initialization
    rospy.init_node('pr2', anonymous=True)

    pr2 = PR2(model_file)

    #initialize point cloud subscriber
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pr2.pcl_callback, queue_size=1)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except:
        pass
```

## 2. Perception Pipeline
 ### a. Filtering and Clustering
  In order to interpret the point cloud message, significant processing was required. I tackled this problem by making separate methods for all of the processing steps that are required in the pick and place project, then calling them in the appropriate order and with experimentally determined parameters in order to achieve the desired result: accurate segmentation of object clusters.
  
  The order of operations implemented in `PR2.segment_scene()` is as follows:
   * Initialize the `Segmenter()` class
   * Convert the ROS point cloud message to point cloud library format
   * Downsample the point cloud to a more manageable density using `Segmenter.voxel_grid_downsample(cloud, leaf_size)`
   * Remove outliers from the point cloud using `Segmenter.outlier_filter(cloud, n_neighbors, threshhold_scale)`
   * Apply a passthrough filter to remove points below the table and the front edge of the table using `Segmenter.axis_passthrough_filter(self, cloud, axis, bounds)`
   * Use RANSAC plane segmentation to separate the table surface points from object points using `Segmenter.ransac_plane_segmentation(cloud, max_distance)`
   * Apply a secondary outlier filter to the object clouds
   * Separate the objects cloud into separate object clouds with Euclidean clustering using `Segmenter.get_euclidean_cluster_indices(self, cloud, tolerance, size_bounds)`
   * Detect objects with the pre-trained SVM model using `Segmenter.detect_objects(cloud, cluster_indices)`
   * Publish the results of the object detection to RViz labels using `Segmenter.publish_detected_objects(detected_objects, marker_pub, objects_pub)`
   * Set class variables with the detection results for later use
   
  Note: some lines are removed/modified in `segment_scene()` for clarity.
  ```python
  def segment_scene(self, pcl_msg):
        seg = self.segmenter #to reduce verbosity below

        # Convert ROS msg to PCL data
        cloud = ros_to_pcl(pcl_msg)
        leaf_size = 0.005

        # Voxel Grid Downsampling
        cloud = seg.voxel_grid_downsample(cloud, leaf_size = leaf_size)

        # Reduce outlier noise in object cloud
        cloud = seg.outlier_filter(cloud, 15, 0.01)
        # Save this to publish to RViz
        denoised_cloud = cloud

        # Passthrough Filter
        cloud = seg.axis_passthrough_filter(cloud, 'z', (0.55, 2)) #filter below table
        cloud = seg.axis_passthrough_filter(cloud, 'x', (.35, 10)) #filter out table front edge

        # RANSAC Plane Segmentation
        # Extract inliers and outliers
        table_cloud, objects_cloud = seg.ransac_plane_segmentation(cloud, max_distance = leaf_size)

        #Reduce outlier noise in object cloud
        objects_cloud = seg.outlier_filter(objects_cloud, 10, 0.01)

        # Euclidean Clustering and Object Detection
        cluster_indices = seg.get_euclidean_cluster_indices(objects_cloud, 0.03, (10,5000))
        detected_objects, detected_objects_dict = seg.detect_objects(objects_cloud, cluster_indices)
        
        # Convert PCL data to ROS messages
        # Publish ROS messages
        message_pairs = [(denoised_cloud, self.denoised_pub),
                         (objects_cloud, self.objects_pub)
                         ]
        
        seg.convert_and_publish(message_pairs)

        #publish detected objects and labels
        seg.publish_detected_objects(detected_objects,
                                     self.object_markers_pub,
                                     self.detected_objects_pub)

        self.object_list = detected_objects_dict
        self.detected_objects = detected_objects
        self.table_cloud = table_cloud
   ```

 ### b. Feature Extraction and Object Detection SVM
  I chose to make the sensor stick package a separate repo pulled into the pick and place project repo as a submodule. This helped me avoid copying it into the perception project or having duplicated training/capture functions for the perception exercises and project.
  
  Within sensor_stick, I updated the feature extraction script (sensor_stick/src/sensor_stick/features.py) used to train my object detection SVM in a few key ways:
   1. Added YCbCr color histogram extraction to my feature vector
   2. Switched from RGB to HSV color histogram extraction
   3. Updated the number of histogram bins (normals, HSV color, and YCbCr color) to 16 instead of 32
 
  My features.py, train_svm.py, and capture_features.py files can be found within the sensor_stick submodule here:
   * [features.py](https://github.com/croomjm/RoboND-Perception-Exercises_sensor_stick/blob/upstream-sensor_stick/src/sensor_stick/features.py)
   * [capture_features.py](https://github.com/croomjm/RoboND-Perception-Exercises_sensor_stick/blob/upstream-sensor_stick/scripts/capture_features.py)
   * [train_svm.py](https://github.com/croomjm/RoboND-Perception-Exercises_sensor_stick/blob/upstream-sensor_stick/scripts/train_svm.py)
   
  I saw dramatic improvement in the efficacy of my object detection model by using both HSV and YCbCr color spaces even using half as many histogram bins. Both of these color spaces allow for the separation of brightness from the color of the object, which helps remove ambiguity due to the position/orientation of an object relative to a light source (unlike RGB). In order to compensate for the addition of an additional histogram calculation, I reduced the number of bins in each histogram by half. To my surprise, this did not have a significant negative effect on the accuracy of my object detection model, though it did significantly increase the speed with which the histograms could be calculated.
  
  The relevant functions are:
  ```python
  def compute_color_histograms(cloud):
       # Compute histograms for the clusters
       point_colors_list = []
       points = pc2.read_points(cloud, skip_nans=True)

       # Step through each point in the point cloud
       for point in points:
           rgb_list = float_to_rgb(point[3])
           point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
           point_colors_list.append(rgb_to_YCbCr(rgb_list))

       normed_features = return_normalized_hist(point_colors_list, bins_range = (0,256))

       return normed_features
  
  def rgb_to_hsv(rgb_list):
       rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
       hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
       return hsv_normalized

  def rgb_to_YCbCr(rgb_list):
       r, g, b = rgb_list

       Y = 16 + (65.738*r + 129.057*g + 25.064*b)/256.
       Cb = 128 +  (-37.945*r - 74.203*g + 112.0*b)/256.
       Cr = 128 + (112.0*r - 93.786*g + 18.214*b)/256.

       YCbCr = [Y, Cb, Cr]

       return YCbCr
   
   def return_normalized_hist(features, bins_range, nbins = 16):
       hist = []
       features = np.asarray(features)
       length, depth = features.shape

       for i in range(depth):
           hist.extend(np.histogram(features[:,i], bins = nbins, range = bins_range)[0])

       hist = hist/np.sum(hist).astype(np.float)

       return hist
   ```
   
   I ultimate trained my object detection SVM using 100 random orientations of each object. After some experimentation, I chose to use a sigmoid kernel because it seemed to generate the most repeatable and accurate object labels. My confusion matrix results are shown below:
   
   ![Normalized Confusion Matrix][normalized_confusion_matrix]

## 3. Successful Object Detection in Each Test World
 My object detection SVM was relatively successful in identifying the objects in the 3 test worlds. I achieved the following correctly identified object detection rates:
  * Test World 1 ([see yaml](./output_1.yaml)): 3 of 3 (100%)
  * Test World 2 ([see yaml](./output_2.yaml)): 5 of 5 (100%)
  * Test World 3 ([see yaml](./output_3.yaml)): 7 of 8 (88%)
 
 Here are the camera views as captured in RViz:
 
  Test World 1
  
  ![Test World 1][test_world_1_result]
  
  Test World 2
  
  ![Test World 2][test_world_2_result]
  
  Test World 3
  
  ![Test World 3][test_world_3_result]

## 4. Successful Pick and Place!
 With the collision map in place and a fairly accurate object detection SVM, I was able to successfully pick and place objects! Here's a video of a successful run:
 
 [![Successful Pick and Place Run](https://img.youtube.com/vi/bGgx0UMarA0/0.jpg)](https://www.youtube.com/watch?v=bGgx0UMarA0)
 
 Unfortunately, since I had trouble updating the collision map after each successive object was picked, I was only able to pick and place a single object with the collision map enabled. With the collision map disabled, I was able to pick multiple objects in a row, though there was some uncertaintly in whether they would make it to the drop box intact...
 
 I found that even when the correct grasp location was passed to the pick and place service, repeatability was a serious problem. I also was not able to test out the pick and place function on larger object sets due to performance limitations on my laptop. Gazebo + RViz is quite heavy, especially within the virtual machine (running on only 2 cores).

## 5. Building the Collision Map
 I first chose to build the collision map without rotating the PR2 world joint. I chose to implement the collision map in two stages. First, I built a baseline map for the static objects that would not be grasped. In the simple case in which the robot does not rotate, this includes only the table. Then, each time I detected the next object in the pick list queue, I appended all other objects that had not yet been picked to the collision map cloud. I then published the collision cloud to the `/pr2/3d_map/points` topic. Unfortunately, for reasons I wasn't able to determine, the collision cloud did not update each time I published a new set of points.
 
 ```python
 def build_collision_map(self):
    # update this to include drop box object cloud as well
    # would need to train SVM for this
    self.collision_map_base_list.extend(list(self.table_cloud))

 def publish_collision_map(self,picked_objects):
     obstacle_cloud_list = self.collision_map_base_list

     for obj in self.detected_objects:
         if obj.label not in picked_objects:
             print('Adding {0} to collision map.'.format(obj.label))
             obj_cloud = ros_to_pcl(obj.cloud)
             obstacle_cloud_list.extend(list(obj_cloud))
             #print(obstacle_cloud)
         else:
             print('Not adding {0} to collision map because it appears in the picked object list'.format(obj.label))

      obstacle_cloud = pcl.PointCloud_PointXYZRGB()
      obstacle_cloud.from_list(obstacle_cloud_list)

      self.segmenter.convert_and_publish([(obstacle_cloud, self.collision_cloud_pub)]) 
   ```
   
   I was also able to implement a method to rotate the robot to look to the left and right at the side tables and drop boxes. I had planned on adding these to the base collision map, but I ran into a couple of issues. First, I hadn't originally trained the SVM to recognize the drop boxes, so I didn't have a quick way to recognize them and add them to the collison map. In addition, I realized that my assumptions in the segment scene method (e.g. distance to the front of the table for plane segmentation) may not be accurate for the side tables. In any case, once I saw that the paths generated by the pick and place service never really came close to the drop box, I decided there wasn't really a need to add the side table and drop box to the collision map.
   
   Here's my code for moving the robot, though... Each time the main script starts up, the robot will move itself according to `PR2.goal_positions`, an array of world joint orientations at which to observe the obstacles in the environment. Once all the goal positions have been achieved and the collision map observed for that orientation, the primary arm mover method is called to pick and place the objects.
   
   ```python
   if len(self.goal_positions)>0:
       new_position = self.goal_positions.pop()
       self.move_world_joint(new_position)

       #segment scene and detect objects
       self.segment_scene(pcl_msg)

       #add obstacles (except for recognized pick objects)
       #to the base collision map
       self.build_collision_map()
   else:
       #identify the objects listed in the pick list
       #submit them to the pick_place_routine
       self.mover()
   ```
   
   Here are the methods that actually move the robot joint.
   
   ```python
   def move_world_joint(self, goal):
       print('Attempting to move world joint to {}'.format(goal))
       pub_j1 = rospy.Publisher('/pr2/world_joint_controller/command',
                                Float64, queue_size=10)
       increments = 10
       position = self.get_world_joint_state()

       goal_positions = [n*1.0/increments*(goal-position) + position for n in range(1, increments + 1)]
       for i, g in enumerate(goal_positions):
           print('Publishing goal {0}: {1}'.format(i, g))
           while abs(position - g) > .005:
               pub_j1.publish(g)
               position = self.get_world_joint_state()
               print('Position: {0}, Error: {1}'.format(position, abs(position - g)))
               rospy.sleep(1)
    
     def get_world_joint_state(self):
       try:
           msg = rospy.wait_for_message("joint_states", JointState, timeout = 10)
           index = msg.name.index('world_joint')
           position = msg.position[index]
       except rospy.ServiceException as e:
           print('Failed to get world_joint position.')
           position = 10**6

       return position
   ``` 
## 6. Bloopers:
 I noticed quite a few issues with the implementation of the pick and place robot project. For one, the simulation is really heavy, and I had a lot of trouble getting it to run successfully in the VM on my laptop (Macbook Air). I got around the issue by working in test world 1 for the most part since having fewer objects seemed to reduce the computation load significatly. However, when trying to rotate the robot, I routinely got down to <2 fps in the simulation.
 
 I also had problems getting the environment to launch successfully. A number of times I had to completely restart my VM after numerous unsuccessful attempts just to launch the program.
 
 Here's an example of one of the repeated issues along those lines. I'm not sure what gave it the idea, but it seems like PR2 was trying to make shadow puppets:
 
 ![Put a bird on it!][shadow_puppets]
 
 Dramatic reenactment:
 
 ![Colbert??](https://media.giphy.com/media/l46Cs9TmyCLhpVHsA/giphy.gif)
 
## 7. Potential Improvements:
 There are still a number of things I would like to improve if I were to continue working on this project.
 
 1. Get a better computer:
  Not really a technical improvement, but I think I would have had a much easier time with a more powerful computer...
 2. Improve the collision map:
  I'd like to retrain my SVM to recognize the drop boxes so I could add them to the static collision map and get a fuller represenation of the map by rotating the robot around. In addition, I never figured out why the collision map wasn't updating in RViz even though I was sending it a new map with the next object to pick removed from the map.
 3. Change the PCL callback structure:
  There are some inherent limitations in the way the pcl callback method is written. It gets a single point cloud, which it uses to locate the objects in the scene and generate a collision map. Unfortunately, since the pick and place service isn't completely reliable, some objects may be moved, dropped, etc., resulting in inaccurate collision maps and grasp locations. I'd like to consider other workflows that would allow me to check whether the object was successfully dropped in the box, update the collision map and grasp locations if the objects move, and reattempt failed pick and place operations.
 4. Improve feature extraction efficiency:
  I realized that a significant portion of the computational load comes from generating the feature histograms for all of the object clusters in the scene. Buiding the histogram seems to be fairly inefficient process in Numpy, but it is made much more so by not vectorizing the operation. I didn't dig into it too much, but Numpy's `numpy.histogramdd()` seems like it would be much more efficient than an iterative approach. 


|  | Learning Rate | Batch Size | Epochs | Steps per Epoch | Validation Steps | Optimizer | IOU | Score | Dataset |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.005 | 256 | 12 | 500 | 50 | Nadam | 0.562 | 0.411 | Supplemental |
| 2 | 0.005 | 64 | 15 | 400 | 50 | Nadam | 0.535 | 0.405 | Baseline |
| 3 | 0.005 | 64 | 12 | 400 | 50 | Nadam | 0.541 | 0.398 | Supplemental |
| 4 | 0.005 | 128 | 15 | 400 | 50 | Adam | 0.537 | 0.392 | Baseline |
| 5 | 0.01 | 256 | 15 | 500 | 50 | Nadam | 0.529 | 0.390 | Baseline |
| 6 | 0.005 | 64 | 15 | 400 | 50 | Adam | 0.539 | 0.389 | Baseline |
| 7 | 0.002 | 64 | 15 | 500 | 50 | Adam | 0.535 | 0.386 | Baseline |
| 8 | 0.01 | 256 | 15 | 500 | 50 | Adam | 0.518 | 0.377 | Baseline |
| 9 | 0.005 | 128 | 12 | 400 | 50 | Nadam | 0.520 | 0.374 | Supplemental |
| 10 | 0.01 | 128 | 15 | 500 | 50 | Adam | 0.506 | 0.370 | Baseline |
| 11 | 0.01 | 64 | 15 | 500 | 50 | Adam | 0.505 | 0.363 | Baseline |
| 12 | 0.002 | 128 | 15 | 500 | 50 | Adam | 0.501 | 0.361 | Baseline |


|  | Overall<br>Score | Following Target<br>Percent False Negatives | No Target<br>Percent False Positives | Far from Target<br>Percent False Positives | Far from Target<br>Percent False Negatives | Dataset |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.411 | 0.0% | 27.306% | 0.619% | 52.632% | Supplemental |
| 2 | 0.405 | 0.0% | 16.236% | 0.31% | 52.941% | Baseline |
| 3 | 0.398 | 0.0% | 26.199% | 0.929% | 51.703% | Supplemental |
| 4 | 0.392 | 0.0% | 18.819% | 0.31% | 58.204% | Baseline |
| 5 | 0.39 | 0.0% | 12.915% | 0.619% | 60.062% | Baseline |
| 6 | 0.389 | 0.0% | 23.985% | 0.929% | 57.276% | Baseline |
| 7 | 0.386 | 0.0% | 28.413% | 0.31% | 55.108% | Baseline |
| 8 | 0.377 | 0.0% | 16.974% | 0.619% | 60.372% | Baseline |
| 9 | 0.374 | 0.0% | 19.188% | 0.31% | 60.991% | Supplemental |
| 10 | 0.37 | 0.0% | 10.332% | 0.31% | 63.158% | Baseline |
| 11 | 0.363 | 0.184% | 16.236% | 0.31% | 62.539% | Baseline |
| 12 | 0.361 | 0.0% | 13.653% | 0.31% | 64.396% | Baseline |
