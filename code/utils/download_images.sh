wget -P ~/RoboND-DeepLearning-Project/data/ https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip
wget -P ~/RoboND-DeepLearning-Project/data/ https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip
wget -P ~/RoboND-DeepLearning-Project/data/ https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip

unzip ~/RoboND-DeepLearning-Project/data/train.zip -d ~/RoboND-DeepLearning-Project/data
unzip ~/RoboND-DeepLearning-Project/data/validation.zip -d ~/RoboND-DeepLearning-Project/data
unzip ~/RoboND-DeepLearning-Project/data/sample_evaluation_data.zip -d ~/RoboND-DeepLearning-Project/data

rm ~/RoboND-DeepLearning-Project/data/train.zip
rm ~/RoboND-DeepLearning-Project/data/validation.zip 
rm ~/RoboND-DeepLearning-Project/data/sample_evaluation_data.zip

mv ~/RoboND-DeepLearning-Project/data/train_combined ~/RoboND-DeepLearning-Project/data/train

echo "Do you wish to download supplemental training data?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) ~/RoboND-DeepLearning-Project/code/utils/download_supplemental_training_data.sh; break;;
        No ) exit;;
    esac
done

echo "Do you wish to download supplemental validation data?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) ~/RoboND-DeepLearning-Project/code/utils/download_supplemental_validation_data.sh; break;;
        No ) exit;;
    esac
done


