import time
import csv
import os

LOG_FILE = "logs/inference_logs.csv"

def log_inference(duration_ms):
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "latency_ms"])
        writer.writerow([time.time(), duration_ms])
