"""DigitForge - handwritten digit reconstruction with RBM and VAE.

Run:  streamlit run app.py
"""
import streamlit as st

from sections import analytics, comparison, reconstruction
from utils.device import device_info
from utils.paths import ensure_dirs
from utils.ui import ensure_trained, get_engine

st.set_page_config(page_title="DigitForge", page_icon="🔢", layout="wide")
ensure_dirs()


def home():
    st.title("🔢 DigitForge")
    st.markdown(
        "Upload an image of handwritten digits. DigitForge finds each digit, normalises it to a 28×28 MNIST-style "
        "image, reconstructs it with an **RBM** and a **VAE** (trained locally on MNIST), compares the two, and then "
        "shows measured performance graphs."
    )
    info = device_info()
    st.caption(f"Running on **{info['device']}** ({info['name']}).")
    ensure_trained()
    engine = get_engine()

    reconstruction.render(engine)          # 1. upload -> detect -> reconstruct
    if st.session_state.get("recon_results") is None:
        st.stop()
    st.divider()
    comparison.render(engine)              # 2. model comparison
    st.divider()
    analytics.render(engine)               # 3. graphs


pages = [
    st.Page(home, title="Home", icon="🏠", default=True, url_path="home"),
    st.Page("pages/About.py", title="About the Models", icon="📖"),
]
st.navigation(pages).run()
