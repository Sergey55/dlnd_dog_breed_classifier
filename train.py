from pytorch_lightning import Trainer
from argparse import ArgumentParser

from util import create_data_module, create_model, create_config

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(args):
    config = create_config()
    model = create_model(args.mode)
    dm = create_data_module(config)

    print(f'Training for {args.max_epochs} epochs')
    
    # Output training parameters
    config.display()

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--mode', default='scratch', help='Model type: scratch, densenet or efficientnet')

    parser = Trainer.add_argparse_args(parser)

    main(parser.parse_args())