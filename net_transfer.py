import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision.models import densenet121

class DogsBreedClassifierTransfer(pl.LightningModule):

    def __init__(self):
        super(DogsBreedClassifierTransfer, self).__init__()

        self.densenet = densenet121(pretrained=True)

        for param in self.densenet.parameters():
            param.requires_grad = False

        self.densenet.classifier = nn.Linear(1024, 133)

    def forward(self, x):
        return self.densenet(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.densenet.classifier.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.TrainResult(loss)

        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)

        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.EvalResult()
        result.log('test_loss', loss)

        return result        


