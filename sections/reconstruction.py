import io
import zipfile


def render(engine):
    import numpy as np
    import pandas as pd
    import streamlit as st

    from utils.digit_detection import draw_boxes, extract_digits
    from utils.metrics import METRICS
    from utils.paths import MODEL_LABELS, MODEL_NAMES
    from utils.plotting import png_bytes, to_pil
    from utils.preprocessing import SegmentationParams, load_image, to_grayscale
    from utils.reconstruction import reconstruct_digits
    from utils.ui import get_engine, require_models, sidebar_status

    st.header("1 · Upload and reconstruct")
    if not engine.available:
        require_models(engine)
        st.stop()

    # ------------------------------------------------------------------ input
    source = st.radio("Input source", ["Upload image", "Camera"], horizontal=True)
    gray = None
    if source == "Upload image":
        up = st.file_uploader("PNG, JPG or JPEG containing handwritten digit(s)", type=["png", "jpg", "jpeg"])
        if up is not None:
            try:
                gray = to_grayscale(load_image(up.getvalue()))
            except Exception as e:
                st.error(f"Could not read this image: {e}")
    elif source == "Camera":
        cam = st.camera_input("Take a photo of a digit on plain paper")
        if cam is not None:
            gray = to_grayscale(load_image(cam.getvalue()))

    params = SegmentationParams()  # automatic background detection, Otsu threshold, MNIST-style normalisation

    digits: list[np.ndarray] = []
    if gray is None:
        st.info("Provide an image to begin.")
        st.stop()
    ex = extract_digits(gray, params)  # always automatic detection
    st.caption(f"Background detected as {'light' if ex.info['light_background'] else 'dark'}"
               + (f", threshold {ex.info['threshold']:.0f}" if ex.info["threshold"] is not None else "") + ".")
    if not ex.boxes:
        st.warning("No digits found in this image. Try a clearer image with dark digits on plain paper.")
    else:
        keep = st.multiselect("Digits to use (deselect false detections)", list(range(1, len(ex.boxes) + 1)),
                              default=list(range(1, len(ex.boxes) + 1)))
        max_n = st.number_input("Maximum number of digits to process", 1, 50, min(len(ex.boxes), 20))
        sel = [i - 1 for i in keep][: int(max_n)]
        a, b = st.columns([3, 2])
        a.image(draw_boxes(gray, ex.boxes, set(sel)), caption="Uploaded image with detected digits", width="stretch")
        b.image(ex.mask, caption="Foreground mask", width="stretch")
        digits = [ex.digits[i] for i in sel]

    if not digits:
        st.stop()

    st.subheader("Extracted digits (28×28, MNIST format)")
    size = st.slider("Display size (px)", 60, 300, 140, 10, key="size_reconstruction")
    cols = st.columns(min(len(digits), 8))
    for i, d in enumerate(digits):
        cols[i % len(cols)].image(to_pil(d, size), caption=f"Digit {i + 1}")

    # ------------------------------------------------------------------ reconstruction controls
    st.subheader("Reconstruction")
    c1, c2, c4 = st.columns(3)
    choice = c1.selectbox("Models", ["Compare both", *[MODEL_LABELS[n] for n in MODEL_NAMES]])
    default_steps = engine.configs["rbm"].infer_steps if "rbm" in engine.configs else 1
    steps = c2.number_input("RBM Gibbs steps", 1, 500, int(st.session_state.get("rbm_infer_steps", default_steps)))
    seed = c4.number_input("Random seed", 0, 10**6, 0)
    models = list(MODEL_NAMES) if choice.startswith("Compare") else [n for n in MODEL_NAMES if MODEL_LABELS[n] == choice]

    if not require_models(engine, models):
        st.stop()

    if st.button("Reconstruct", type="primary"):
        with st.spinner("Reconstructing..."):
            st.session_state["digits"] = digits
            st.session_state["recon_settings"] = {"rbm_steps": int(steps), "vae_stochastic": False,
                                                  "seed": int(seed)}
            st.session_state["recon_results"] = reconstruct_digits(
                engine, digits, models, int(steps), False, int(seed))
            st.session_state["recon_models"] = models

    results = st.session_state.get("recon_results")
    if results is None:
        st.stop()
    models = st.session_state["recon_models"]
    rs = st.session_state["recon_settings"]
    if "rbm" in models:
        st.caption(f"RBM used **{rs['rbm_steps']}** Gibbs sampling step(s). "
                   f"VAE decodes the latent mean (deterministic).")

    hdr = st.columns(1 + len(models))
    hdr[0].markdown("**Original**")
    for c, n in zip(hdr[1:], models):
        c.markdown(f"**{MODEL_LABELS[n]}**")
    for i, r in enumerate(results):
        row = st.columns(1 + len(models))
        row[0].image(to_pil(r["original"], size), caption=f"Digit {i + 1}")
        for c, n in zip(row[1:], models):
            c.image(to_pil(r["recon"][n], size), caption=f"MSE {r['metrics'][n]['MSE']:.4f}")

    st.subheader("Metrics vs. the preprocessed upload")
    rows = [{"Digit": i + 1, "Model": MODEL_LABELS[n], "MSE": r["metrics"][n]["MSE"], "MAE": r["metrics"][n]["MAE"]}
            for i, r in enumerate(results) for n in models]
    st.dataframe(pd.DataFrame(rows).style.format({"MSE": "{:.4f}", "MAE": "{:.4f}"}),
                 hide_index=True, width="stretch")
    st.caption("These measure pixel similarity to the preprocessed input. An uploaded image has no clean "
               "ground truth, so they do **not** prove the reconstructed digit is semantically the same digit. "
               "Lower is better for both.")

    # ------------------------------------------------------------------ downloads
    st.subheader("Download")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, r in enumerate(results, 1):
            z.writestr(f"digit{i}_original.png", png_bytes(r["original"]))
            for n in models:
                z.writestr(f"digit{i}_{n}.png", png_bytes(r["recon"][n]))
    st.download_button("Download all images (ZIP)", buf.getvalue(), "digitforge_reconstructions.zip", "application/zip")
    dl = st.columns(len(results) * len(models) if len(results) * len(models) <= 6 else 6)
    k = 0
    for i, r in enumerate(results, 1):
        for n in models:
            dl[k % len(dl)].download_button(f"Digit {i} · {MODEL_LABELS[n]}", png_bytes(r["recon"][n]),
                                            f"digit{i}_{n}.png", "image/png", key=f"dl_{i}_{n}")
            k += 1

    # ------------------------------------------------------------------ VAE interpolation
    with st.expander("VAE latent-space interpolation"):
        if "vae" not in engine.models:
            st.warning("VAE not available.")
        elif len(digits) < 2:
            st.info("Needs at least two extracted digits (e.g. upload `201`).")
        else:
            a = st.selectbox("From digit", range(1, len(digits) + 1), 0)
            b = st.selectbox("To digit", range(1, len(digits) + 1), 1)
            n_steps = st.slider("Interpolation steps", 3, 12, 8)
            frames = engine.interpolate_vae(digits[a - 1], digits[b - 1], n_steps)
            cols = st.columns(n_steps)
            for j, f in enumerate(frames):
                cols[j].image(to_pil(f, 90), caption=f"t={j / (n_steps - 1):.2f}")
