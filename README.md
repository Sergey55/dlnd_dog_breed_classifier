[//]: # (Image References)

[image1]: ./images/result.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"

# Dog's breed classifier

## Project Overview

The Convolutional Neural Network (CNN) project for classifying dog's breed. In this project I've implemented a pipeline that can be used within a web or mobile app. Given an image of a dog, the network will identify an estimate of the canine's breed.  

![Sample Output][image1]

## Data

#TODO:

## Network

#TODO:

### Training

#TODO:

### Evaluation

#TODO:

## Files

* (Classifier.ipynb)[./Classifier.ipynb]
* (environment.yml)[./environment.yml]
* (main.py)[./main.py]
* (net_transfer_densenet121.py)[./net_transfer_densenet121.py]
* (net_transfer_EfficientNet.py)[./net_transfer_EfficientNet.py]
* (net.py)[./net.py]
* (Training Colab.ipynb)[./Training Colab.ipynb]

## Results

Finally I've got three models which can be used for classifying canine's breed by photo.

Short sammary in table below:

|---------------|---------------|---------------|-----------|-----------|
|Model			|Accuracy		|F1 Score		|Presicio	|Recal		|
|---------------|---------------|---------------|-----------|-----------|
|Scratch		|29.7%			|0.259			|0.276		|0.276		|
|DenseNet121	|79.4%			|0.796			|0.819		|0.780		|
|EfficientNet	|68.5%			|0.663			|0.734		|0.674		|
|---------------|---------------|---------------|-----------|-----------|