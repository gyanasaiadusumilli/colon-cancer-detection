"""Streamlit demo: upload a colon histopathology image and get a prediction.

Run:
    streamlit run app.py
"""
import cv2
import numpy as np
import streamlit as st

from src.data import apply_clahe
from src.evaluate import load_checkpoint
from src.predict import predict_image
from src.utils import get_device

st.set_page_config(page_title="Colon Cancer Detection", layout="centered")
st.title("Colon Cancer Detection from Histopathology Images")
st.caption("Research demo only. Not a medical device and not for clinical use.")


@st.cache_resource
def get_model(path="outputs/best_model.pt"):
    device = get_device()
    model, ckpt = load_checkpoint(path, device)
    return model, ckpt, device


uploaded = st.file_uploader("Upload an H&E stained colon tissue image", type=["jpg", "jpeg", "png"])
if uploaded:
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    model, ckpt, device = get_model()

    col1, col2 = st.columns(2)
    col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    col2.image(cv2.cvtColor(apply_clahe(img), cv2.COLOR_BGR2RGB), caption="After CLAHE",
               use_container_width=True)

    result = predict_image(img, model, ckpt, device)
    st.subheader(f"Prediction: {result['label']}")
    st.metric("Confidence", f"{result['confidence']:.2%}")
    st.bar_chart(result["probabilities"])
