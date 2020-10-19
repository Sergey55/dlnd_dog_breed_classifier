[//]: # (Image References)

[image1]: ./images/result.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"

# Dog's breed classifier

## Project Overview

The Convolutional Neural Network (CNN) project for classifying dog's breed. In this project I've implemented a pipeline that can be used within a web or mobile app. Given an image of a dog, the network will identify an estimate of the canine's breed.  

![Sample Output][image1]

## Data

This project uses the [Dogs dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). It contains photos of 133 different dog breeds.

Archive with photos supposed to be downloaded and extracted to the `dog_images` folder. The `dog_images` folder should contain three subfolders with train, validation, and test datasets accordingly. Each subfolder should contain 133 subdirectories, each corresponding to a different dog breed.

```
project_home
|
├── dog_images
|   |
|   ├── train
|   |   ├── 001.Affenpinscher
|   |   ├── 002.Afghan_hound
|   |   ├── ...
|   |   └── 133.Yorkshire_terrier
|   |
|   ├── test
|   |   ├── 001.Affenpinscher
|   |   ├── 002.Afghan_hound
|   |   ├── ...
|   |   └── 133.Yorkshire_terrier
|   |
|   ├── valid
|   |   ├── 001.Affenpinscher
|   |   ├── 002.Afghan_hound
|   |   ├── ...
|   |   └── 133.Yorkshire_terrier
...
```

## Network

#TODO:

### Training

#TODO:

### Evaluation

#TODO:

## Files

* [Classifier.ipynb](./Classifier.ipynb) - Evaluating logic
* [environment.yml](./environment.yml) - Conda environment info
* [train.py](./train.py) - Training on local machine
* [test.py](./test.py) - Tesing on local machine
* [net_transfer_densenet121.py](./net_transfer_densenet121.py) - Model implementation using transfer learning based on DenseNet121
* [net_transfer_EfficientNet.py](./net_transfer_EfficientNet.py) - Model implementation using transfer learning based on EfficientNetB6
* [net.py](./net.py) - Scratch model iplementation
* [kgs.txt](./pkgs.txt) - List of required packages
* [Training_Colab.ipynb](./Training_Colab.ipynb)

## Results

Finally I've got three models which can be used for classifying canine's breed by photo.

Short sammary in table below:

|Model			|Accuracy		|F1 Score		|Presicio	|Recal		|
|---------------|---------------|---------------|-----------|-----------|
|Scratch		|29.7%			|0.259			|0.276		|0.276		|
|DenseNet121	|79.4%			|0.796			|0.819		|0.780		|
|EfficientNet	|68.5%			|0.663			|0.734		|0.674		|