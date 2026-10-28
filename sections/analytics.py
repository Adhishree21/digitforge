import numpy as np
import pandas as pd
import streamlit as st


def render(engine):
    from utils import plotting as P
    from utils.evaluation import evaluate, load_evaluation
    from utils.paths import MODEL_LABELS
    from utils.ui import get_engine, sidebar_status, theme_name

    st.header("3 · Graphs")
    theme = theme_name()

    if not engine.available:
        st.error("No trained models found.")
        st.stop()
    for n, e in engine.errors.items():
        st.warning(f"{MODEL_LABELS[n]} excluded: {e}")

    # ------------------------------------------------------------------ measure once, reuse until checkpoints change
    steps = int(st.session_state.get("recon_settings", {}).get("rbm_steps", 1))
    result, arrays, problem = load_evaluation(engine)
    if result is not None and (result["rbm_steps"] != steps or result["n_samples"] != 10000):
        result = None  # stale (other Gibbs steps or subset) -> re-measure
    if result is None:
        bar = st.progress(0.0, "Measuring all models on the 10,000 held-out MNIST test images...")
        evaluate(engine, 10000, steps, progress=lambda f: bar.progress(f, "Measuring on held-out MNIST test images..."))
        bar.empty()
        result, arrays, problem = load_evaluation(engine)
    st.caption(f"Measured on {result['n_samples']} held-out MNIST test images on {result['device']} "
               f"(RBM Gibbs steps = {result['rbm_steps']}). These graphs describe the models, not your upload.")

    # ------------------------------------------------------------------ Graph 2 (only needs checkpoints)
    st.header("Graph 2 · Training convergence")
    hist = {n: engine.histories[n] for n in engine.available}
    fig = P.fig_convergence(hist, theme)
    st.pyplot(fig)
    st.caption("Each objective is model-specific, so absolute values are **not** comparable across panels: the RBM "
               "panel shows CD reconstruction error (CD does not minimise it exactly); the VAE shows the negative ELBO "
               "split into its parts. Plot titles and axes differ per model on purpose.")

    models = [m for m in ("rbm", "vae") if m in result["summary"]]
    S = result["summary"]


    def best(metric, higher):
        f = max if higher else min
        return f(models, key=lambda m: S[m][metric]["mean"])


    # ------------------------------------------------------------------ Graph 1
    st.header("Graph 1 · Reconstruction quality (held-out test set)")
    st.pyplot(P.fig_quality_bars(S, theme))
    tbl = pd.DataFrame({MODEL_LABELS[m]: {k: S[m][k]["mean"] for k in ("MSE", "MAE")} for m in models})
    st.dataframe(tbl.style.format("{:.4f}"), width="stretch")
    st.caption("Bars = mean over test images, whiskers = standard deviation. Lower is better for both. "
               "Measured: " + "; ".join(
                   f"{k} best = {MODEL_LABELS[best(k, h)]}" for k, h in (("MSE", False), ("MAE", False))) + ".")

    # ------------------------------------------------------------------ Graph 3
    st.header("Graph 3 · Inference time")
    st.pyplot(P.fig_latency(result["latency"], theme))
    st.caption(result["latency_method"])

    # ------------------------------------------------------------------ Graph 4
    st.header("Graph 4 · Error distribution")
    st.pyplot(P.fig_error_distribution(arrays, models, theme))
    med = {m: float(np.median(arrays[f"{m}__MSE"])) for m in models}
    p95 = {m: float(np.percentile(arrays[f"{m}__MSE"], 95)) for m in models}
    st.caption("Median MSE: " + ", ".join(f"{MODEL_LABELS[m]} {med[m]:.4f}" for m in models) +
               ". 95th percentile: " + ", ".join(f"{MODEL_LABELS[m]} {p95[m]:.4f}" for m in models) +
               ". A long right tail means occasional badly reconstructed digits.")

    # ------------------------------------------------------------------ Graph 6
    st.header("Graph 5 · Model complexity")
    st.pyplot(P.fig_complexity(result["complexity"], result["latency"], theme))
    cx = result["complexity"]
    st.dataframe(pd.DataFrame({MODEL_LABELS[m]: {"Parameters (inference path)": cx[m]["params_inference"],
                                                 "Parameters (all components)": cx[m]["params_total"],
                                                 "Checkpoint (MB)": round(cx[m]["checkpoint_mb"], 2)} for m in models}),
                 width="stretch")
    st.caption("RBM parameters are counted although they are updated by Contrastive Divergence rather than by "
               "gradient descent.")
