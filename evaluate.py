"""
Evaluation script.

Can be used for obtaining prediction of dog's breed by a photo.

Example of usage: 
    python evaluate.py --mode densenet --checkpoint_path './logs/transfer_densenet_logs/lightning_logs/version_1/checkpoints/epoch=4.ckpt' --image_path './dog_images/test/085.Irish_red_and_white_setter/Irish_red_and_white_setter_05766.jpg'

Written by Sergey Andreev
"""
import json

import torch
from torchvision.transforms import ToTensor

from pytorch_lightning import Trainer
from argparse import ArgumentParser

from util import create_model, create_config

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(args):
    with open('classes.json', 'r') as f:
        classes = json.load(f)
    
    # Convert key values to ints
    classes = {int(k): v for k, v in classes.items()}

    model = create_model(args.mode, path_to_checkpoint=args.checkpoint_path)

    model.eval()

    image = Image.open(args.image_path)
    tensor = ToTensor()(image).unsqueeze(0)
    prediction = torch.argmax(model(tensor), dim=1).item()

    print(f'Predicted class: {classes[prediction]}')

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--mode', required=True, help='Model type: scratch, densenet or efficientnet')
    parser.add_argument('--checkpoint_path', required=True, help='Path to checkpoint')
    parser.add_argument('--image_path', required=True, help='Path to the image file to be classified')

    main(parser.parse_args())