from fastapi import FastAPI
import torch
from src.generator import Generator

app = FastAPI(title="Synthetic Image Generator API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator(100).to(device)
G.load_state_dict(torch.load("outputs/G_final.pt", map_location=device))
G.eval()

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/generate")
def generate_images(num_images: int = 16):
    z = torch.randn(num_images, 100).to(device)
    with torch.no_grad():
        images = G(z).cpu().tolist()
    return {"images": images}
