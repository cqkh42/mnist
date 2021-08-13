import torchvision
import torch
from torch.utils.data import DataLoader


def load_mnist():
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x: torch.flatten(x))]
    )
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transforms)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)
    return train_dl, test_dl


def load_linear(m=-0.5, c=0, n=50):
    X = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    noise = torch.randn((n, 1))
    y = (m * X) + noise + c
    return DataLoader(list(zip(X, y)), batch_size=n, shuffle=True)
