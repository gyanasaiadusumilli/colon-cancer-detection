"""Streamlit demo: classify a colon histopathology image as benign or adenocarcinoma.

Run locally:
    streamlit run app.py

If outputs/best_model.pt is missing, the trained model, its test metrics and the sample
images are downloaded from the Hugging Face model repo named in MODEL_REPO.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.data import apply_clahe
from src.evaluate import load_checkpoint
from src.predict import predict_image
from src.utils import get_device

CKPT = Path("outputs/best_model.pt")
METRICS = Path("outputs/metrics.json")
EXAMPLES = Path("examples")
MODEL_REPO = os.environ.get("MODEL_REPO", "Gyana23/colon-cancer-resnet18")

st.set_page_config(page_title="Colon Cancer Detection", page_icon="🔬", layout="centered")


@st.cache_resource(show_spinner="Downloading the trained model...")
def _download_assets():
    # Cached only when it succeeds; a failed download raises and is retried on the next visit
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_REPO, local_dir=".", allow_patterns=["outputs/*", "examples/*"])


def fetch_assets():
    """Download the model, metrics and sample images once, if they are not already here."""
    if not CKPT.exists():
        try:
            _download_assets()
        except Exception as e:  # show a clear message instead of crashing
            st.warning(f"Could not download the model from {MODEL_REPO}: {e}")


fetch_assets()
st.title("Colon Cancer Detection")
st.write(
    "Upload an H&E stained colon tissue image, or pick a sample below. "
    "The model enhances contrast with OpenCV (CLAHE), then a fine-tuned ResNet-18 "
    "predicts whether the tissue is benign or adenocarcinoma."
)
st.caption("Research demo only. Not a medical device and not for clinical use.")

if METRICS.exists():
    m = json.loads(METRICS.read_text())
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{m['accuracy']:.1%}")
    c2.metric("F1 score", f"{m['f1']:.3f}")
    c3.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    st.caption(f"Measured on {m['n_test']:,} held-out images from the LC25000 dataset.")


@st.cache_resource
def get_model():
    device = get_device()
    model, ckpt = load_checkpoint(str(CKPT), device)
    return model, ckpt, device


if not CKPT.exists():
    st.error("The model isn't available yet. Train it with notebooks/train_on_colab.ipynb, "
             "or set MODEL_REPO to a Hugging Face repo that contains outputs/best_model.pt.")
    st.stop()

img = None
samples = sorted(EXAMPLES.glob("*.jp*g")) + sorted(EXAMPLES.glob("*.png")) if EXAMPLES.exists() else []
if samples:
    st.subheader("Try a sample")
    cols = st.columns(len(samples))
    for col, path in zip(cols, samples):
        col.image(str(path))
        if col.button(path.stem.replace("_", " ").title(), key=path.name):
            st.session_state["sample"] = str(path)

uploaded = st.file_uploader("Or upload your own image", type=["jpg", "jpeg", "png"])
if uploaded:
    st.session_state.pop("sample", None)
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
elif "sample" in st.session_state:
    img = cv2.imread(st.session_state["sample"])

if img is not None:
    model, ckpt, device = get_model()
    result = predict_image(img, model, ckpt, device)

    st.divider()
    col1, col2 = st.columns(2)
    col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input")
    col2.image(cv2.cvtColor(apply_clahe(img), cv2.COLOR_BGR2RGB), caption="After CLAHE enhancement")

    if result["label"] == "Adenocarcinoma":
        st.error(f"Prediction: **{result['label']}** ({result['confidence']:.1%} confidence)")
    else:
        st.success(f"Prediction: **{result['label']}** ({result['confidence']:.1%} confidence)")
    st.bar_chart(result["probabilities"], horizontal=True)

st.divider()
st.markdown("[Source code on GitHub](https://github.com/gyanasaiadusumilli/colon-cancer-detection) · "
            "Built by [Gyanasai Adusumilli](https://www.linkedin.com/in/gyanasai)")
