import streamlit as st
import torch
import torchvision
import numpy as np
import os
import zipfile
from generator import Generator

import cv2

def upscale_nearest(img, scale=8):
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Synthetic Image Generator (Vanilla GAN)",
    layout="wide"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("⚙️ Controls")

num_images = st.sidebar.slider("Number of images", 1, 64, 16)
seed = st.sidebar.number_input("Random Seed (optional)", value=0, step=1)

generate_btn = st.sidebar.button("Generate Images")

# --------------------------------------------------
# Main Title
# --------------------------------------------------
st.title("Synthetic Image Generator (Vanilla GAN)")
st.markdown(
    """
This application generates **privacy-preserving synthetic images**
using a **Vanilla Generative Adversarial Network (GAN)**.

**Use cases**
- Data augmentation
- Model robustness testing
- Privacy-safe dataset sharing
- Educational demonstrations
"""
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(100).to(device)
    model.load_state_dict(torch.load("outputs/G_final.pt", map_location=device))
    model.eval()
    return model, device

# --------------------------------------------------
# Image Generation
# --------------------------------------------------
if generate_btn:
    st.subheader("Generated Images")

    with st.spinner("Generating synthetic images..."):
        if seed != 0:
            torch.manual_seed(seed)

        G, device = load_model()
        z = torch.randn(num_images, 100).to(device)

        with torch.no_grad():
            images = G(z).cpu()

        # Save individual images
        os.makedirs("samples/generated", exist_ok=True)
        for i, img in enumerate(images):
            torchvision.utils.save_image(
                img,
                f"samples/generated/img_{i}.png",
                normalize=True,
                value_range=(-1, 1)
            )
        cols = st.columns(4)
        for i, img in enumerate(images):
            
            img_np = img.squeeze().numpy()
            img_np = upscale_nearest(img_np)
            cols[i % 4].image(img_np, clamp=True, caption=f"Img {i}", width=100)

        # Create grid for display
        grid = torchvision.utils.make_grid(
            images, nrow=4, normalize=True, value_range=(-1, 1)
        )
        grid_np = grid.permute(1, 2, 0).numpy()

    st.image(grid_np, caption="Synthetic Images (28×28)", width="content")

    st.success("✅ Images generated successfully!")

    # --------------------------------------------------
    # ZIP Download
    # --------------------------------------------------
    zip_path = "samples/generated_images.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for i in range(num_images):
            zipf.write(f"samples/generated/img_{i}.png")

    with open(zip_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Images as ZIP",
            data=f,
            file_name="synthetic_images.zip",
            mime="application/zip"
        )
