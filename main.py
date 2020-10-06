import os
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import random_split
from datamodule import DogsDataModule

from net import DogsBreedClassifier

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    
    model = DogsBreedClassifier()

    paths = {
        'train': './dog_images/train/',
        'test': './dog_images/test/',
        'val': './dog_images/valid/'
    }

    dm = DogsDataModule(paths)

    trainer = pl.Trainer(
        max_epochs=50
    )
    trainer.fit(model, dm)

    trainer.test(datamodule=dm)

if __name__ == '__main__':
    main()