wget -P ~/RoboND-DeepLearning-Project/data/ https://s3-us-west-2.amazonaws.com/robond-deep-learn-project-training-data/additional_train.zip
wget -P ~/RoboND-DeepLearning-Project/data/ https://s3-us-west-2.amazonaws.com/robond-deep-learn-project-training-data/additional_train.zip

unzip ~/RoboND-DeepLearning-Project/data/additional_train.zip -d ~/RoboND-DeepLearning-Project/data
unzip ~/RoboND-DeepLearning-Project/data/additional_train2.zip -d ~/RoboND-DeepLearning-Project/data

rm ~/RoboND-DeepLearning-Project/data/*.zip

echo "Number of files in train folder:"
ls ~/RoboND-DeepLearning-Project/data/train/images/ -1 | wc -l

echo "Number of files in additional_train folder:"
ls ~/RoboND-DeepLearning-Project/data/additional_train/images/ -1 | wc -l

echo "Number of files in additional_train2 folder:"
ls ~/RoboND-DeepLearning-Project/data/additional_train2/images/ -1 | wc -l

mv ~/RoboND-DeepLearning-Project/data/additional_train/images/* ~/RoboND-DeepLearning-Project/data/train/images/
mv ~/RoboND-DeepLearning-Project/data/additional_train/masks/* ~/RoboND-DeepLearning-Project/data/train/masks/
mv ~/RoboND-DeepLearning-Project/data/additional_train2/images/* ~/RoboND-DeepLearning-Project/data/train/images/
mv ~/RoboND-DeepLearning-Project/data/additional_train2/masks/* ~/RoboND-DeepLearning-Project/data/train/masks/
rm -r ~/RoboND-DeepLearning-Project/data/additional_train/
rm -r ~/RoboND-DeepLearning-Project/data/additional_train2/

echo "Number of files in train folder:"
ls ~/RoboND-DeepLearning-Project/data/train/images/ -1 | wc -l