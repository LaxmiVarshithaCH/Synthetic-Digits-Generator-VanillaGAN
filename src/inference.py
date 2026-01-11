import torch
import torchvision
import os
from generator import Generator

def generate_images(
    model_path="outputs/G_final.pt",
    num_images=16,
    noise_dim=100,
    output_dir="samples"
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(noise_dim).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    z = torch.randn(num_images, noise_dim).to(device)

    with torch.no_grad():
        images = G(z).cpu()

    grid = torchvision.utils.make_grid(
        images, nrow=4, normalize=True, value_range=(-1, 1)
    )

    out_path = os.path.join(output_dir, "inference.png")
    torchvision.utils.save_image(grid, out_path)

    print(f"[INFO] Inference image saved at {out_path}")

if __name__ == "__main__":
    generate_images()
