import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class DogsDataModule(pl.LightningDataModule):

    def __init__(self, data_paths: dict, batch_size = 32, num_workers = 8):
        super().__init__()
        self.data_paths = data_paths
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        train_dataset = ImageFolder(root=self.data_paths['train'], transform=self.create_train_transformation())

        return DataLoader(
            train_dataset,
            batch_size = self.batch_size,
            shuffle=True,
            num_workers = self.num_workers)

    def val_dataloader(self):
        validation_dataset = ImageFolder(root=self.data_paths['val'], transform=self.create_validation_transformation())

        return DataLoader(
            validation_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers)

    def test_dataloader(self):
        test_dataset = ImageFolder(root=self.data_paths['test'], transform=self.create_test_transformation())

        return  DataLoader(
            test_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers)

    def create_normalization_transformation(self):
        """Create normalization transformation common for all datasets

        Returns:
            Instance of torchvision.transforms.Normalize class"""

        return transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))

    def create_train_transformation(self) -> torchvision.transforms.Compose:
        """Create appropriate transformation for train dataset

        Returns:
            Instance of torchvision.transforms.Compose class"""

        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.create_normalization_transformation()])

    def create_validation_transformation(self) -> torchvision.transforms.Compose:
        """Create appropriate transformation for validation dataset

        Returns:
            Instance of torchvision.transforms.Compose class"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.create_normalization_transformation()])

    def create_test_transformation(self) -> torchvision.transforms.Compose:
        """Create appropriate transformation for validation dataset

        Returns:
            Instance of torchvision.transforms.Compose class"""

        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.create_normalization_transformation()])