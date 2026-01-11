import torch.nn as nn
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator

def build_gan(cfg, device):
    G = Generator(cfg["noise_dim"]).to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], 0.999))
    opt_D = optim.Adam(D.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], 0.999))

    return G, D, criterion, opt_G, opt_D
