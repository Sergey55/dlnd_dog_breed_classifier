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
        model = DogsBreedClassifier.load_from_checkpoint(args.checkpoint_path)
    elif args.mode == 'densenet':
        print('Using `DenseNet121` network')
        model = DogsBreedClassifierDenseNet.load_from_checkpoint(args.checkpoint_path)
    else:
        print('Using `EfficientNet` network')
        model = DogsBreedClassifierEfficientNet.load_from_checkpoint(args.checkpoint_path)

    paths = {
        'train': './dog_images/train/',
        'test': './dog_images/test/',
        'val': './dog_images/valid/'
    }

    dm = DogsDataModule(paths)

    trainer = Trainer.from_argparse_args(args)
    trainer.test(datamodule=dm, model=model)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--mode', default='scratch', help='Model type: scratch, densenet or efficientnet')
    parser.add_argument('--checkpoint_path', required=True, help='Path to checkpoint')

    main(parser.parse_args())