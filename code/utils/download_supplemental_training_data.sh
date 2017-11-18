wget -P ~/RoboND-DeepLearning-Project/data/ https://s3-us-west-2.amazonaws.com/robond-deep-learn-project-training-data/additional_train.zip

unzip ~/RoboND-DeepLearning-Project/data/additional_train.zip -d ~/RoboND-DeepLearning-Project/data

rm ~/RoboND-DeepLearning-Project/data/*.zip

mv ~/RoboND-DeepLearning-Project/data/additional_train/images/* ~/RoboND-DeepLearning-Project/data/train/images/
mv ~/RoboND-DeepLearning-Project/data/additional_train/masks/* ~/RoboND-DeepLearning-Project/data/train/masks/