import fastai.vision.all as vision
import torch
import torchvision
import torch
from torch.utils.data import DataLoader


def load_mnist():
    # transforms = torchvision.transforms.Compose(
    #     [torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x: torch.flatten(x))]
    # )
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True)

    X_train = (
            train_dataset.data[(train_dataset.targets == 3) | (train_dataset.targets == 7)]
            .view(-1, 28 * 28)
            .float()
            / 255
    )
    y_train = train_dataset.targets[(train_dataset.targets == 3) | (train_dataset.targets == 7)].unsqueeze(-1)
    y_train = (y_train == 3).float()
    train_dl = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=256, shuffle=True)

    X_test = test_dataset.data[(test_dataset.targets == 3) | (test_dataset.targets == 7)].view(-1,
                                                                                               28 * 28).float() / 255
    y_test = test_dataset.targets[(test_dataset.targets == 3) | (test_dataset.targets == 7)].unsqueeze(-1)
    y_test = (y_test == 3).float()
    test_dl = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=256, shuffle=True)
    return train_dl, test_dl


def load_linear(m=-0.5, c=0, n=50):
    X = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    noise = torch.randn((n, 1))
    y = (m * X) + noise + c
    return DataLoader(list(zip(X, y)), batch_size=n, shuffle=True)
