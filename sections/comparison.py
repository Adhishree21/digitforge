import numpy as np
import pandas as pd
import streamlit as st


def render(engine):
    from utils.metrics import METRICS, rank_models
    from utils.paths import MODEL_LABELS, MODEL_NAMES
    from utils.plotting import to_pil
    from utils.reconstruction import reconstruct_digits
    from utils.ui import get_engine, require_models, sidebar_status

    st.header("2 · Model comparison")

    digits = st.session_state.get("digits")
    if not digits:
        st.info("No digits yet. Upload an image above first.")
        st.stop()
    if not require_models(engine):
        st.stop()

    rs = st.session_state.get("recon_settings", {"rbm_steps": 1, "vae_stochastic": False, "seed": 0})
    size = st.slider("Display size (px)", 60, 300, 140, 10, key="size_comparison")
    results = reconstruct_digits(engine, digits, list(MODEL_NAMES), rs["rbm_steps"], False, rs["seed"])
    st.caption(f"Settings from the reconstruction step: RBM Gibbs steps = {rs['rbm_steps']}, "
               f"VAE deterministic (latent mean), seed = {rs['seed']}.")

    hdr = st.columns(3)
    for c, t in zip(hdr, ["Original", *[MODEL_LABELS[n] for n in MODEL_NAMES]]):
        c.markdown(f"**{t}**")
    for i, r in enumerate(results):
        row = st.columns(3)
        row[0].image(to_pil(r["original"], size), caption=f"Digit {i + 1}")
        for c, n in zip(row[1:], MODEL_NAMES):
            m = r["metrics"][n]
            c.image(to_pil(r["recon"][n], size), caption=f"MSE {m['MSE']:.4f} · SSIM {m['SSIM']:.3f}")


    def ranking_table(metric_values: dict[str, dict[str, float]]) -> pd.DataFrame:
        rows = []
        for metric, (_, higher) in METRICS.items():
            vals = metric_values[metric]
            order = rank_models(vals, higher)
            rows.append({"Metric": metric, "Better when": "higher" if higher else "lower",
                         **{f"#{k + 1}": f"{MODEL_LABELS[m]} ({vals[m]:.4f})" for k, m in enumerate(order)}})
        return pd.DataFrame(rows)


    st.subheader("Ranking per metric (average over all processed digits)")
    avg = {mt: {n: float(np.mean([r["metrics"][n][mt] for r in results])) for n in MODEL_NAMES} for mt in METRICS}
    st.dataframe(ranking_table(avg), hide_index=True, width="stretch")

    if len(results) > 1:
        with st.expander("Ranking for each individual digit"):
            for i, r in enumerate(results, 1):
                st.markdown(f"**Digit {i}**")
                st.dataframe(ranking_table({mt: {n: r["metrics"][n][mt] for n in MODEL_NAMES} for mt in METRICS}),
                             hide_index=True, width="stretch")

    winners = {mt: MODEL_LABELS[rank_models(avg[mt], h)[0]] for mt, (_, h) in METRICS.items()}
    if len(set(winners.values())) == 1:
        st.success(f"All four metrics favour {next(iter(winners.values()))} for these digits.")
    else:
        st.warning("Different metrics favour different models here (" +
                   ", ".join(f"{m}: {w}" for m, w in winners.items()) + "). No single overall winner is declared.")
    st.caption("Rankings use pixel/structure similarity to the preprocessed upload only; they say nothing about "
               "whether a reconstruction is still recognisable as the same digit.")
