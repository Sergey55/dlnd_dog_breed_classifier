import os

from pytorch_lightning import Trainer
from argparse import ArgumentParser

from datamodule import DogsDataModule

from net import DogsBreedClassifier
from net_transfer_densenet121 import DogsBreedClassifierDenseNet
from net_transfer_EfficientNet import DogsBreedClassifierEfficientNet

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(args):
    
    if args.mode == 'scratch':
        print('Using `Scratch` model')
        model = DogsBreedClassifier()
    elif args.mode == 'densenet':
        print('Using `DenseNet121` network')
        model = DogsBreedClassifierDenseNet()
    else:
        print('Using `EfficientNet` network')
        model = DogsBreedClassifierEfficientNet()

    paths = {
        'train': './dog_images/train/',
        'test': './dog_images/test/',
        'val': './dog_images/valid/'
    }

    dm = DogsDataModule(paths)

    print(f'Training for {args.max_epochs} epochs')

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--mode', default='scratch', help='Model type: scratch, densenet or efficientnet')

    parser = Trainer.add_argparse_args(parser)

    main(parser.parse_args())