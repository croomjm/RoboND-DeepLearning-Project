wget -P ~/RoboND-DeepLearning-Project/data/ https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip
wget -P ~/RoboND-DeepLearning-Project/data/ https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip
wget -P ~/RoboND-DeepLearning-Project/data/ https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip

unzip ~/RoboND-DeepLearning-Project/data/train.zip
unzip ~/RoboND-DeepLearning-Project/data/validation.zip
unzip ~/RoboND-DeepLearning-Project/data/sample_evaluation_data.zip

rm ~/RoboND-DeepLearning-Project/data/*.zip

mv ~/RoboND-DeepLearning-Project/data/train_combined ~/RoboND-DeepLearning-Project/data/train
