import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision.models import densenet121
from torchvision.models import resnext101_32x8d

from efficientnet_pytorch import EfficientNet

class DogsBreedClassifierDenseNet(pl.LightningModule):

    def __init__(self):
        super(DogsBreedClassifierDenseNet, self).__init__()

        self.model = densenet121(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Linear(1024, 133)

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.classifier.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # result = pl.TrainResult(loss)

        acc = self.train_acc(y_hat, y)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        self.log('training_acc_epoch', self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        acc = self.val_acc(y_hat, y)
        self.log('val_acc_step', acc, on_step=True, on_epoch=False)

        return {'val_loss': loss}

    def valication_epoch_end(self, outputs):
        self.log('val_acc_epoch', self.val_acc.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.EvalResult()
        result.log('test_loss', loss)

        return result

class DogsBreedClassifierEfficientNet(pl.LightningModule):

    def __init__(self):
        super(DogsBreedClassifierEfficientNet, self).__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=133)

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # result = pl.TrainResult(loss)

        acc = self.train_acc(y_hat, y)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        self.log('training_acc_epoch', self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        acc = self.val_acc(y_hat, y)
        self.log('val_acc_step', acc, on_step=True, on_epoch=False)

        return {'val_loss': loss}

    def valication_epoch_end(self, outputs):
        self.log('val_acc_epoch', self.val_acc.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        return {'test_loss': loss}        


