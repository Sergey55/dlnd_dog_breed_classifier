"""
Base Configuration class

Written by Sergey Andreev
"""

class Config(object):
    """Configuration class
    
    Create a sub-class that inherits from this one and override properties
    that need to be changed."""

    # Path to train dataset
    TRAIN_DATA_DIR = './dog_images/train/'

    # Path to validation dataset
    VALID_DATA_DIR = './dog_images/test/'

    # Path to test dataset
    TEST_DATA_DIR = './dog_images/valid/'

    def __init__(self):
        pass

    def display(self):
        """Display configuration values"""
        print('\nConfiguration:')

        for setting in dir(self):
            if not setting.startswith('__') and not callable(getattr(self, setting)):
                print(f'{setting}: {getattr(self, setting)}')