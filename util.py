"""
Utility methods

Written by Sergey Andreev
"""
from pytorch_lightning import LightningDataModule, LightningModule
from datamodule import DogsDataModule

from net_scratch import DogsBreedClassifier
from net_transfer import DogsBreedClassifierDenseNet, DogsBreedClassifierEfficientNet

from config import Config

def create_config() -> Config:
    return Config()

def create_data_module(config: Config) -> LightningDataModule:
    """Create instance of LightningDataModule class which can be
    used for creating DataLoader for particular dataset.

    Returns:
        Instance of LightningDataModule
    """
    paths = {
        'train': config.TRAIN_DATA_DIR,
        'test': config.TEST_DATA_DIR,
        'val': config.VALID_DATA_DIR
    }

    return DogsDataModule(paths)

def create_model(mode: str, path_to_checkpoint = None) -> LightningModule:
    """Create instance of model of desired type

    Args:
        mode: Model type. One of ['scratch', 'densenet', 'efficientnet']
        path_to_checkpoint: (Optional) Load model weight from file
    Returns:
        Instance of LightningModule"""

    assert mode != None and mode != ''

    if mode == 'scratch':
        if path_to_checkpoint != None:
            model = DogsBreedClassifier.load_from_checkpoint(path_to_checkpoint)
        else:
            model = DogsBreedClassifier()
    elif mode == 'densenet':
        if path_to_checkpoint != None:
            model = DogsBreedClassifierDenseNet.load_from_checkpoint(path_to_checkpoint)
        else:
            model = DogsBreedClassifierDenseNet()
    else:
        if path_to_checkpoint != None:
            model = DogsBreedClassifierEfficientNet.load_from_checkpoint(path_to_checkpoint)
        else:
            model = DogsBreedClassifierEfficientNet()

    return model