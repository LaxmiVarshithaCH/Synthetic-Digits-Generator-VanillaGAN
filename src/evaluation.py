import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from generator import Generator
from utils.metrics import extract_features, calculate_fid, get_inception_model

# --------------------------------------------------
# Evaluation Script (Memory-Safe)
# --------------------------------------------------

def main():
    os.makedirs("figures", exist_ok=True)

    device = torch.device("cpu")  # FORCE CPU (Mac-safe)

    # -------------------------------
    # Load config
    # -------------------------------
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # -------------------------------
    # Load trained Generator
    # -------------------------------
    G = Generator(cfg["noise_dim"]).to(device)
    G.load_state_dict(torch.load("outputs/G_final.pt", map_location=device))
    G.eval()

    # -------------------------------
    # Load real dataset
    # -------------------------------
    transform = transforms.Compose([
        transforms.Resize(cfg["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    real_dataset = datasets.MNIST(
        root="data/raw",
        train=True,
        download=True,
        transform=transform
    )

    real_loader = DataLoader(
        real_dataset,
        batch_size=32,   # 🔻 reduced
        shuffle=True
    )

    # -------------------------------
    # Collect limited samples
    # -------------------------------
    MAX_SAMPLES = 200   # 🔻 critical fix

    real_imgs = []
    fake_imgs = []

    with torch.no_grad():
        for imgs, _ in real_loader:
            imgs = imgs.to(device)
            z = torch.randn(imgs.size(0), cfg["noise_dim"]).to(device)
            fake = G(z)

            real_imgs.append(imgs.cpu())
            fake_imgs.append(fake.cpu())

            if sum(x.size(0) for x in real_imgs) >= MAX_SAMPLES:
                break

    real_imgs = torch.cat(real_imgs, dim=0)[:MAX_SAMPLES]
    fake_imgs = torch.cat(fake_imgs, dim=0)[:MAX_SAMPLES]

    # -------------------------------
    # Feature extraction (Inception)
    # -------------------------------
    inception = get_inception_model(device)

    rf = extract_features(real_imgs, inception, device)
    ff = extract_features(fake_imgs, inception, device)

    # -------------------------------
    # Metrics
    # -------------------------------
    fid = calculate_fid(rf, ff)
    diversity = float(np.mean(np.std(ff, axis=0)))

    print(f"FID Score: {fid:.2f}")
    print(f"Diversity Score: {diversity:.4f}")

    # -------------------------------
    # t-SNE visualization
    # -------------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=20,
        random_state=42,
        init="random"
    )

    emb = tsne.fit_transform(np.vstack([rf, ff]))

    plt.figure(figsize=(6, 6))
    plt.scatter(emb[:len(rf), 0], emb[:len(rf), 1], s=8, label="Real")
    plt.scatter(emb[len(rf):, 0], emb[len(rf):, 1], s=8, label="Fake")
    plt.legend()
    plt.title("t-SNE: Real vs Fake")
    plt.savefig("figures/tsne_real_vs_fake.png")
    plt.close()

    print("t-SNE plot saved to figures/tsne_real_vs_fake.png")


if __name__ == "__main__":
    main()
