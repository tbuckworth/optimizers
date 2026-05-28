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


def get_mnist_small_loaders(n_train=500, batch_size=64, seed=42, data_dir="./data"):
    """Small MNIST subset — heavily overparameterized setting for overfitting.
    OOD: test images with Gaussian noise."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    rng = np.random.RandomState(seed)
    per_class = n_train // 10
    indices = []
    for c in range(10):
        class_idx = (train_ds.targets == c).nonzero(as_tuple=True)[0].numpy()
        chosen = rng.choice(class_idx, per_class, replace=False)
        indices.extend(chosen.tolist())
    rng.shuffle(indices)
    train_subset = torch.utils.data.Subset(train_ds, indices)

    test_raw = test_ds.data.float().unsqueeze(1) / 255.0
    test_raw = (test_raw - 0.1307) / 0.3081
    test_labels = test_ds.targets

    ood_rng = np.random.RandomState(seed + 100)
    noise = torch.tensor(ood_rng.randn(*test_raw.shape).astype(np.float32)) * 0.5
    ood_images = test_raw + noise

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    ood_loader = DataLoader(TensorDataset(ood_images, test_labels),
                            batch_size=256, shuffle=False)
    return train_loader, test_loader, ood_loader


def get_noisy_spiral_loaders(n_train=400, train_noise=0.8, batch_size=64, seed=42):
    """Noisy spirals — memorization-prone setting.
    OOD: points at larger radii (extrapolation)."""
    X_train, y_train = make_spirals(n_samples=n_train, noise=train_noise, seed=seed)
    X_test, y_test = make_spirals(n_samples=400, noise=0.3, seed=seed + 1)

    rng = np.random.RandomState(seed + 2)
    n_ood = 200
    points_per_class = n_ood // 2
    X_ood_list, y_ood_list = [], []
    for c in range(2):
        r = np.linspace(1.2, 2.0, points_per_class)
        theta = np.linspace(c * np.pi, c * np.pi + 2.5 * np.pi, points_per_class)
        theta += rng.randn(points_per_class) * 0.3 * 0.3
        x1 = r * np.cos(theta) + rng.randn(points_per_class) * 0.05
        x2 = r * np.sin(theta) + rng.randn(points_per_class) * 0.05
        X_ood_list.append(np.column_stack([x1, x2]))
        y_ood_list.append(np.full(points_per_class, c))
    X_ood = torch.from_numpy(np.vstack(X_ood_list).astype(np.float32))
    y_ood = torch.from_numpy(np.concatenate(y_ood_list).astype(np.int64))

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=256, shuffle=False)
    ood_loader = DataLoader(TensorDataset(X_ood, y_ood),
                            batch_size=256, shuffle=False)
    return train_loader, test_loader, ood_loader


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
