import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size=64, random_labels=False, seed=42, data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    if random_labels:
        rng = np.random.RandomState(seed)
        train_ds.targets = torch.tensor(rng.randint(0, 10, len(train_ds.targets)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader


def make_spirals(n_samples=1000, noise=0.3, seed=42, n_classes=2):
    rng = np.random.RandomState(seed)
    points_per_class = n_samples // n_classes
    X, y = [], []
    for c in range(n_classes):
        r = np.linspace(0.2, 1.0, points_per_class)
        theta = np.linspace(c * 2 * np.pi / n_classes,
                            c * 2 * np.pi / n_classes + 2.5 * np.pi,
                            points_per_class) + rng.randn(points_per_class) * noise * 0.3
        x1 = r * np.cos(theta) + rng.randn(points_per_class) * noise * 0.05
        x2 = r * np.sin(theta) + rng.randn(points_per_class) * noise * 0.05
        X.append(np.column_stack([x1, x2]))
        y.append(np.full(points_per_class, c))
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def get_spiral_loaders(batch_size=64, random_labels=False, seed=42):
    X_train, y_train = make_spirals(n_samples=2000, noise=0.3, seed=seed)
    X_test, y_test = make_spirals(n_samples=500, noise=0.3, seed=seed + 1)
    # OOD: rotated by 45 degrees
    angle = np.pi / 4
    rot = torch.tensor([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]], dtype=torch.float32)
    X_ood = X_test @ rot.T
    y_ood = y_test

    if random_labels:
        rng = np.random.RandomState(seed)
        y_train = torch.from_numpy(rng.randint(0, 2, len(y_train)).astype(np.int64))

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=256, shuffle=False)
    ood_loader = DataLoader(TensorDataset(X_ood, y_ood),
                            batch_size=256, shuffle=False)
    return train_loader, test_loader, ood_loader
