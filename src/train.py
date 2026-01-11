import yaml, torch, os
from data_loader import get_dataloaders
from vanilla_gan import build_gan
from utils.visualizer import save_samples
from utils.logger import init_logger, log_epoch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

train_loader, _ = get_dataloaders()
G, D, criterion, opt_G, opt_D = build_gan(cfg, device)

writer = SummaryWriter()
init_logger()

for epoch in range(1, cfg["epochs"] + 1):
    for real, _ in train_loader:
        real = real.to(device)
        bsz = real.size(0)

        real_labels = torch.full((bsz,1), 0.9).to(device)
        fake_labels = torch.zeros(bsz,1).to(device)

        # Discriminator
        opt_D.zero_grad()
        d_real = criterion(D(real), real_labels)

        z = torch.randn(bsz, cfg["noise_dim"]).to(device)
        fake = G(z)
        d_fake = criterion(D(fake.detach()), fake_labels)

        d_loss = d_real + d_fake
        d_loss.backward()
        opt_D.step()

        # Generator
        opt_G.zero_grad()
        g_loss = criterion(D(fake), real_labels)
        g_loss.backward()
        opt_G.step()

    log_epoch(epoch, d_loss.item(), g_loss.item())
    writer.add_scalar("Loss/D", d_loss, epoch)
    writer.add_scalar("Loss/G", g_loss, epoch)

    if epoch % 10 == 0:
        save_samples(G, cfg["noise_dim"], device, epoch)
        torch.save(G.state_dict(), f"checkpoints/G_epoch_{epoch}.pt")

torch.save(G.state_dict(), "outputs/G_final.pt")
torch.save(D.state_dict(), "outputs/D_final.pt")
