import yaml
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch

def get_dataloaders():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    transform = transforms.Compose([
        transforms.Resize(cfg["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.MNIST(
        root="data/raw",
        train=True,
        download=True,
        transform=transform
    )

    train_size = int(len(dataset) * cfg["train_split"])
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    # After split
    torch.save(train_ds, "data/processed/train/train.pt")
    torch.save(test_ds, "data/processed/test/test.pt")

    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    print("✅ MNIST loaded successfully")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")