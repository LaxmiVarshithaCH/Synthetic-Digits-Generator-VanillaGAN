import os
import torch
import torchvision
import matplotlib.pyplot as plt

def save_samples(G, noise_dim, device, epoch, out_dir="samples"):
    os.makedirs(out_dir, exist_ok=True)

    G.eval()
    z = torch.randn(16, noise_dim).to(device)

    with torch.no_grad():
        images = G(z).cpu()

    grid = torchvision.utils.make_grid(
        images, nrow=4, normalize=True, value_range=(-1, 1)
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"Epoch {epoch}")
    plt.savefig(f"{out_dir}/epoch_{epoch:03d}.png")
    plt.close()

    G.train()
