import fastai
from fastai.vision.all import (
    DataLoaders, Learner, URLs, cnn_learner, resnet18,
    accuracy, ImageDataLoaders, untar_data
)
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F

from mnist.mean_classifier import MeanClassifier, mse, mae
from mnist import sgd, learner, loss
from mnist import data


train_dl, test_dl = data.load_mnist()
dataloaders = DataLoaders(train_dl, test_dl)

regression_dl = data.load_linear()
for loss_func in mse, mae:
    clf = MeanClassifier(loss=loss_func)
    clf.fit(train_dl)
    print(f'Mean Classifier with {loss_func.__name__.upper()} loss had an accuracy of {clf.score(test_dl):.4}')