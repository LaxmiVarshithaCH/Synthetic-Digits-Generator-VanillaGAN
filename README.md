# Synthetic Image Generator (Vanilla GAN)

A compact, easy-to-run implementation of a Vanilla GAN for generating 28×28 grayscale images (MNIST-style). The repo includes training, evaluation, a Streamlit demo app, and a small FastAPI service for programmatic generation.

---

## 🔧 Key features

- Vanilla GAN with simple Conv/ConvTranspose architecture (Generator & Discriminator)
- Training loop with TensorBoard logging and checkpointing
- Inference script to generate image grids and single images
- Evaluation utilities (FID, diversity, t-SNE visualization)
- Streamlit app for interactive generation and FastAPI for programmatic access

---

## 🚀 Quickstart

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

2. Configure hyperparameters in `config.yaml` (image size, batch size, epochs, lr, etc.)

3. Download dataset and run training:

```bash
# download happens automatically when running the data loader
python src/train.py
```

Checkpoints are saved to `checkpoints/` every 10 epochs and the final models to `outputs/`.

4. Run inference to create a grid of generated images:

```bash
python src/inference.py --model-path outputs/G_final.pt --num-images 16
# or simply: python src/inference.py
```

5. (Optional) Run the demo app or API

```bash
# Streamlit UI
streamlit run src/app.py

# FastAPI server
uvicorn src.api:app --reload
```

Open TensorBoard to inspect training logs:

```bash
tensorboard --logdir=runs
```

---

## ⚙️ Configuration

Edit `config.yaml` to change model/hyperparameters (noise_dim, lr, batch_size, epochs, image_size, etc.).

---

## 📁 Repository layout (high level)

- `src/` — implementation (training, model, data loader, inference, evaluation, app and API)
- `config.yaml` — default hyperparameters
- `checkpoints/` — intermediate generator weights
- `outputs/` — final model weights
- `samples/` — generated images and grids
- `figures/` — evaluation visualizations
- `logs/` — training CSV logs and TensorBoard data

---

## 🧪 Evaluation

Run `python src/evaluation.py` to compute FID/diversity scores and generate a t-SNE plot comparing real vs generated samples. The evaluation script is CPU-friendly and limits samples to be memory-safe.

---

## 🧑‍💻 Development notes

- To reproduce results, use the `config.yaml` defaults and train for at least the number of epochs listed there.
- The code assumes you run scripts from the repository root (e.g., `python src/train.py`).
- Check `src/utils` for logging and visualization helpers.

---

## 🙌 Contributing & Support

Contributions are welcome — please open issues or pull requests. If you have a preferred workflow, add a `CONTRIBUTING.md` and link it here.

If you need help, open an issue or contact the maintainer.

---

## 📜 License

If this repository is missing a `LICENSE` file, add one to make the intended license explicit (e.g., MIT, Apache-2.0).

---

## ✨ Acknowledgements

This is a teaching/experimental repo intended for quick experimentation, demos and small experiments (data augmentation, privacy-preserving synthetic data, demos).

