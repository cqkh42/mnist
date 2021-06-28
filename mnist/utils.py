import fastai.vision.all as vision
from PIL import Image
import numpy as np
import torch


def image_to_tensor(img_file):
    img = Image.open(img_file)
    as_array = np.array(img)
    as_tensor = torch.from_numpy(as_array)
    return as_tensor

def load_mnist():
    path = vision.untar_data(vision.URLs.MNIST_SAMPLE)

    train_threes = (path/'train'/'3').ls().sorted()
    train_threes = [image_to_tensor(img) for img in train_threes]
    train_threes = torch.stack(train_threes).float() / 255

    train_sevens = (path/'train'/'7').ls().sorted()
    train_sevens = [image_to_tensor(img) for img in train_sevens]
    train_sevens = torch.stack(train_sevens).float() / 255

    test_threes = (path/'valid'/'3').ls().sorted()
    test_threes = [image_to_tensor(img) for img in test_threes]
    test_threes = torch.stack(test_threes).float() / 255
    test_sevens = (path/'valid'/'7').ls().sorted()
    test_sevens = [image_to_tensor(img) for img in test_sevens]
    test_sevens = torch.stack(test_sevens).float() / 255

    X_test = torch.cat((test_sevens, test_threes)).view(-1, 28*28)
    X_train = torch.cat((train_sevens, train_threes)).view(-1, 28*28)

    y_train = torch.from_numpy(np.array([7] * len(train_sevens) + [3] * len(train_threes)))
    y_test = torch.from_numpy(np.array([7] * len(test_sevens) + [3] * len(test_threes)))  
    return X_train, X_test, y_train, y_test


def load_diabetes():
    X, y = datasets.load_diabetes(return_X_y=True)
    y = sklearn.preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))[:, 0]
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()