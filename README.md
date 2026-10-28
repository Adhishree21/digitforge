# DigitForge — Handwritten Digit Reconstruction with RBM and VAE

DigitForge is a single local **Streamlit** application. You upload an image containing handwritten digits (e.g. `201`),
it detects and normalises each digit to a 28×28 MNIST-style image, reconstructs every digit independently with two
**separately trained** generative models, and compares them with measured metrics and graphs.
No backend server, database or hosted inference is involved; everything runs locally with PyTorch.

| Model | What reconstructs the digit |
|---|---|
| **RBM** (Bernoulli–Bernoulli) | Clamp the binarised digit → infer hidden units → *n* explicit Gibbs steps |
| **VAE** (convolutional) | Encode → latent mean (or a sample) → decode |

## Installation (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

* Python 3.10 or 3.11 is the target. The code also ran on newer versions during development.
* **GPU:** the default PyPI `torch` wheel on Windows is CPU-only. For the RTX 4050 install the CUDA build first, e.g.
  `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`, then the line above.
  The app uses CUDA automatically when available and silently falls back to CPU otherwise (the Home page and sidebar show
  the device in use).
* MNIST is downloaded once by `torchvision` into `data/` the first time you train (needs internet then; afterwards offline).
  No pretrained weights are downloaded — every model is trained here.

## Training

Training settings are **fixed** in `training/configs.py` (RBM: 512 hidden units, CD‑1, 25 epochs; VAE: latent 20, 20 epochs;
batch size 128; all 60 000 MNIST training images; seed 42). There is nothing to configure in the app.

* On the first launch the Home page trains whichever model has no compatible checkpoint (progress bars are shown, MNIST is
  downloaded once) and then continues. Later launches reuse `checkpoints/{rbm,vae}.pt`; nothing is retrained on reruns.
* To retrain, delete the checkpoint file(s) and reload, or run from a terminal (optional flags override the defaults):

```bat
python -m training.train_vae --epochs 20
python -m training.train_rbm --epochs 25 --n_hidden 512 --cd_k 1
```

Checkpoints store weights, the full config and the loss history; the architecture is rebuilt from the saved config so an
incompatible or corrupt file produces a clear error instead of a silent mismatch.

## Using the app

The Home page is one top-to-bottom flow:

1. **Upload** a PNG/JPG/JPEG (RGB, RGBA, grayscale; EXIF orientation is fixed). Check the detected boxes and 28×28 crops;
   deselect any false detections. Preprocessing uses fixed automatic settings.
2. Choose RBM Gibbs steps and seed (the VAE always decodes the latent mean) → **Reconstruct**. One row per digit, a metric table, PNG/ZIP
   downloads and VAE latent interpolation.
3. **Model comparison** appears next: original beside both reconstructions and a per-metric ranking from actual values.
4. **Graphs** appear last: the six measured graphs (computed once on the 10 000 held‑out test images, then cached in
   `outputs/` until a checkpoint changes). The **About the Models** page explains the methods.

## Method

### Preprocessing
Grayscale → polarity detection (border median vs. Otsu threshold; digits end up white on black like MNIST) → Otsu /
adaptive / manual threshold → connected components → merge vertically stacked fragments, drop specks → row/left‑to‑right
ordering → per digit: tight crop, scale to fit 20×20 **preserving aspect ratio**, optional stroke thickening (strokes are
measured as area ÷ skeleton length and thickened to MNIST's ≈2 px before resizing), paste into 28×28, centre by centre of
mass (MNIST convention), scale to [0, 1].

### RBM
Energy `E(v,h) = −bᵀv − cᵀh − vᵀWh`; free energy `F(v) = −bᵀv − Σⱼ softplus(cⱼ + (vᵀW)ⱼ)`. Contrastive Divergence‑k:
`ΔW ∝ ⟨vhᵀ⟩_data − ⟨vhᵀ⟩_k‑step‑Gibbs`, with momentum and weight decay; training images are stochastically binarised. The
mean free energy of train vs. held‑out data is tracked as a diagnostic.
Reconstruction: visible units are clamped to the digit thresholded at 0.5, `p(h|v)` is computed, then `n` Gibbs steps
(`h~p(h|v)`, `v~p(v|h)`); the output is the visible probability map of the last step. The number of steps is shown.

### VAE
`ELBO = E_q[log p(x|z)] − KL(q(z|x) ‖ N(0,I))`. Encoder → `μ, logσ²`; **reparameterisation** `z = μ + σ⊙ε`. Loss =
BCE reconstruction + KL (tracked separately, per image). Reconstruction decodes `μ`; the stochastic mode decodes a sampled `z`.

### Evaluation
* MSE, MAE (lower better), PSNR (higher better, data range 1, capped at 100 dB for identical images), SSIM (higher
  generally better), all between the 28×28 input and the reconstruction.
* **Held‑out test set:** the official 10 000 MNIST test images, never used in training; the same images go to all models.
* **Uploads:** there is no clean ground truth, so metrics compare to the *preprocessed* upload and measure similarity,
  not semantic correctness (a model can output a clean but different digit).
* **Latency:** batch size 1, same input, warm‑up discarded, 100 timed runs with device synchronisation, model loading
  excluded. **Complexity:** parameter counts and checkpoint file sizes read from the loaded models/files.
* Results are cached in `outputs/evaluation.json` + `.npz` and invalidated automatically when a checkpoint changes.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (`checkpoints/` and `outputs/` are included; `data/` is git-ignored).
2. At share.streamlit.io choose the repo, main file `app.py`, and under *Advanced settings* pick Python 3.11 or 3.12.
3. Deploy. `requirements.txt` installs CPU-only PyTorch; no training happens on the server.

## Tests

```bat
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
Covers image loading/EXIF/grayscale, polarity inversion, detection and ordering, 28×28 normalisation, RBM/VAE shapes, CD learning, metric values, ranking direction, checkpoint round‑trip and error handling, tiny end‑to‑end training
and a stop request.

## Project layout

```
app.py                      Home (full flow) + navigation
sections/                   reconstruction, comparison, analytics (the three steps of the Home page)
pages/                      About the Models
models/                     rbm.py, vae.py
training/                   fixed configs, shared helpers, train_{rbm,vae}.py
utils/                      preprocessing, digit_detection, reconstruction, metrics, evaluation, plotting, checkpoints, device, ui
checkpoints/  outputs/  data/  tests/
```

## Screenshots
Run the app and capture them here (not included — none are generated automatically):
`docs/home.png`, `docs/reconstruction.png`, `docs/analytics.png`.

## Known limitations
* Reconstruction does not guarantee the same digit; RBM output especially drifts with many Gibbs steps.
* Digits that touch each other form one connected component and are not split automatically (separate them in the source image).
  Punctuation, minus signs and very short marks are discarded as specks. Multi-line text is read row by row.
* Photos with strong shadows may need *Adaptive* thresholding or the manual threshold.
* The RBM uses a 0.5 binarisation of the input; anti‑aliasing detail is lost for that model only.
* First-run training happens inside the page and blocks it until finished (a few minutes on GPU, longer on CPU).
