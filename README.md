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

In case if a different file structure is required you can always specify it in the `Config` class.

## Network

#TODO:

### Training

All three neural networks trained on Google Colab using [trainig_colab.ipynb](./trainig_colab.ipynb) notebook. All of them can be trained on a local machine as well using the `train.py` script. For e.g. using command below:

```
python train.py --mode densenet
```

Using [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/){:target="_blank"} makes the process way easier. Most of the training parameters can be passed as a key right in terminal. Suppose we want to train our model on GPU during 20 epochs and save logs to a particular directory. We can do it next way:

```
python train.py --mode densenet --gpus 1 --max_epochs 20 --default_root_dir './logs/densenet/`
```

All set of parameter can be obtained using the next command:

```
python train.py -h
```

### Evaluation

Having the model trained prediction can be obtained using [evaluate.py](./evaluate.py) script. Like this:

```
python evaluate.py --mode densenet --checkpoint_path './logs/transfer_densenet_logs/lightning_logs/version_1/checkpoints/epoch=4.ckpt' --image_path './dog_images/test/085.Irish_red_and_white_setter/Irish_red_and_white_setter_05766.jpg'
```

## Files

* [classes.json](./classes.json) - JSON file with a dictionary of all supported classes.
* [Classifier.ipynb](./Classifier.ipynb) - Evaluating logic.
* [config.py](./config.py) - Configuration file.
* [datamodule.py](./datamodule.py) - Pytorch Lightning datamodule which can be used for obtaining dataloader for particulat dataset.
* [environment.yml](./environment.yml) - Conda environment info.
* [net_scratch.py](./net.py) - The model implemented from scratch.
* [net_transfer.py](./net_transfer_densenet121.py) - Models using transfer learning based on DenseNet121 and EfficientNetB6.
* [pkgs.txt](./pkgs.txt) - List of required packages
* [README.md](./README.md) - This file
* [train.py](./train.py) - Script for training network on local machine.
* [training_colab.ipynb](./training_colab.ipynb) - Notebook for training models in Google Colab.
* [util.py](./util.py) - Utility functions.

## Results

Finally I've got three models which can be used for classifying canine's breed by photo.

Short sammary in table below:

|Model			|Accuracy		|F1 Score		|Presicio	|Recal		|
|---------------|---------------|---------------|-----------|-----------|
|Scratch		|29.7%			|0.259			|0.276		|0.276		|
|DenseNet121	|79.4%			|0.796			|0.819		|0.780		|
|EfficientNet	|68.5%			|0.663			|0.734		|0.674		|

The aim of the project is implementing working solution for image classification, however, better results can be obtained using each of the three models.