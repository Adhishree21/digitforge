"""Explanations of the two models and the evaluation method."""
import streamlit as st

from utils.ui import sidebar_status

st.title("📖 About the Models")
sidebar_status()

st.header("Restricted Boltzmann Machine")
st.latex(r"E(v,h)=-b^\top v-c^\top h-v^\top W h,\qquad F(v)=-b^\top v-\sum_j \log\!\left(1+e^{c_j+(v^\top W)_j}\right)")
st.markdown(
    "Binary visible units (binarised pixels) and binary hidden units with no within-layer connections, so "
    "$p(h_j=1|v)=\\sigma(c_j+v^\\top W_j)$ and $p(v_i=1|h)=\\sigma(b_i+W_i h)$. Training uses **Contrastive "
    "Divergence (CD-k)**: $\\Delta W \\propto \\langle v h^\\top\\rangle_{data}-\\langle v h^\\top\\rangle_{k\\text{-step Gibbs}}$.\n\n"
    "**Reconstruction:** the visible units are clamped to the binarised digit, hidden units are inferred, then "
    "*n* explicit Gibbs steps are run; the output is the visible probability map after the last step. More steps drift "
    "further from the input towards the model's own distribution."
)

st.header("Variational Autoencoder")
st.latex(r"\log p(x)\ge \mathbb{E}_{q(z|x)}[\log p(x|z)]-\mathrm{KL}\big(q(z|x)\,\|\,\mathcal N(0,I)\big)")
st.markdown(
    "The encoder outputs $\\mu$ and $\\log\\sigma^2$; the **reparameterisation trick** $z=\\mu+\\sigma\\odot\\epsilon$, "
    "$\\epsilon\\sim\\mathcal N(0,I)$ keeps sampling differentiable. The loss is reconstruction (binary cross-entropy) "
    "plus KL, tracked separately. **Reconstruction:** decode the latent mean (deterministic) or a sampled $z$."
)

st.header("How results are evaluated")
st.markdown(
    "- **MSE, MAE** (lower better), **PSNR** (higher better, dB, data range 1.0), **SSIM** (higher generally better).\n"
    "- On the **held-out MNIST test set** (never used for training) the reference is the true image.\n"
    "- For **uploaded images** there is no clean ground truth; metrics compare against the *preprocessed* 28×28 digit "
    "and measure similarity, not whether the digit is still recognisable.\n"
    "- Latency is measured at batch size 1 after warm-up with device synchronisation; model loading is excluded."
)
