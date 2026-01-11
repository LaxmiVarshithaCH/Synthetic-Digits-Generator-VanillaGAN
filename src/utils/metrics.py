import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm

# -----------------------------
# Inception Model Loader (SAFE)
# -----------------------------
def get_inception_model(device):
    weights = Inception_V3_Weights.IMAGENET1K_V1

    model = inception_v3(weights=weights)
    
    # Disable auxiliary classifier safely
    model.aux_logits = False
    model.AuxLogits = None

    # Replace final FC with Identity → feature extractor
    model.fc = torch.nn.Identity()

    model.to(device)
    model.eval()
    return model

# -----------------------------
# Feature Extraction
# -----------------------------
@torch.no_grad()
def extract_features(images, model, device):
    """
    images: (N, 1, H, W) or (N, 3, H, W)
    """
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    images = F.interpolate(
        images, size=(299, 299),
        mode="bilinear", align_corners=False
    )

    images = images.to(device)
    features = model(images)

    return features.cpu().numpy()

# -----------------------------
# FID
# -----------------------------
def calculate_fid(real_features, fake_features, eps=1e-6):
    mu1 = real_features.mean(axis=0)
    mu2 = fake_features.mean(axis=0)

    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)

    # 🔧 Numerical stability
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2

    covmean = sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


# -----------------------------
# Diversity Score
# -----------------------------
def diversity_score(fake_features):
    return float(np.mean(np.std(fake_features, axis=0)))
