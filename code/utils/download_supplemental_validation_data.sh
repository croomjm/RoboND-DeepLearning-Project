wget -P ~/RoboND-DeepLearning-Project/data/ https://s3-us-west-2.amazonaws.com/robond-deep-learn-project-training-data/additional_validation.zip

unzip ~/RoboND-DeepLearning-Project/data/additional_validation.zip -d ~/RoboND-DeepLearning-Project/data

rm ~/RoboND-DeepLearning-Project/data/*.zip

mv ~/RoboND-DeepLearning-Project/data/additional_validation/images/* ~/RoboND-DeepLearning-Project/data/validation/images/
mv ~/RoboND-DeepLearning-Project/data/additional_validation/masks/* ~/RoboND-DeepLearning-Project/data/validation/masks/
rm -r ~/RoboND-DeepLearning-Project/data/additional_validation/