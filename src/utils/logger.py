import csv
import os

LOG_FILE = "logs/training_logs.csv"

def init_logger():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "D_Loss", "G_Loss"])

def log_epoch(epoch, d_loss, g_loss):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, d_loss, g_loss])
