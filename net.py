import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DogsBreedClassifier(pl.LightningModule):

    def __init__(self):
        super(DogsBreedClassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(7 * 7 * 256, 512)
        self.fc2 = nn.Linear(512, 133)
        
        self.dropout = nn.Dropout(0.25)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fcbn1 = nn.BatchNorm1d(7 * 7 * 256)
        self.fcbn2 = nn.BatchNorm1d(512)

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()        

    def forward(self, x):
        x = self.pool(self.bn1(F.elu(self.conv1(x))))
        x = self.pool(self.bn2(F.elu(self.conv2(x))))
        x = self.pool(self.bn3(F.elu(self.conv3(x))))
        x = self.pool(self.bn4(F.elu(self.conv4(x))))
        
        x = x.view(-1, 7 * 7 * 256)
        
        x = self.dropout(x)
        
        x = self.fcbn1(x)
        
        x = F.elu(self.fc1(x))
        x = self.fcbn2(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

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