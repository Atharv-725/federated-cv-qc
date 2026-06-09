import streamlit as st
import json, os
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(page_title="Federated CV QC Dashboard", layout="wide")
st.markdown("## Federated CV Quality Control Dashboard")
st.markdown("Privacy-preserving defect detection across 3 factories — YOLOv8 + Flower + Differential Privacy")
st.divider()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

log_path = os.path.join(BASE_DIR, "..", "server", "logs", "training_log.json")
if os.path.exists(log_path):
    with open(log_path) as f:
        logs = json.load(f)
    df = pd.DataFrame(logs)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rounds Completed", int(df["round"].max()))
    col2.metric("Total Epsilon Spent", f"{df['epsilon_used'].max():.2f}")
    col3.metric("Clients per Round", int(df["num_clients"].iloc[-1]))
    last_map = df["avg_mAP"].iloc[-1] if "avg_mAP" in df.columns else 0.0
    col4.metric("Final mAP", f"{last_map:.4f}")
    st.subheader("Training Progress")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].plot(df["round"], df["epsilon_used"], color="#534AB7", marker="o", linewidth=2)
    axes[0].set_title("DP Epsilon Budget Consumed per Round")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Cumulative Epsilon")
    axes[0].grid(True, alpha=0.3)
    if "avg_mAP" in df.columns:
        axes[1].plot(df["round"], df["avg_mAP"], color="#1D9E75", marker="o", linewidth=2)
        axes[1].set_title("Global mAP per Round")
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("mAP")
        axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("No training logs found.")

st.divider()
st.subheader("Live Defect Detection")
col_a, col_b = st.columns([1, 2])
with col_a:
    factory = st.selectbox("Select Factory Model", ["factory_a", "factory_b", "factory_c"])
    uploaded = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png"])
with col_b:
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        model_path = os.path.join(BASE_DIR, "..", "runs", "detect", "runs", factory, "train", "weights", "best.pt")
        if os.path.exists(model_path):
            model = YOLO(model_path)
            results = model(img, verbose=False, conf=0.01)
            res_img = results[0].plot()
            st.image(res_img, caption=f"Detection result - {factory}", use_container_width=True)
            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                for box in detections:
                    cls_id = int(box.cls)
                    cls_name = results[0].names[cls_id]
                    conf = float(box.conf)
                    icon = "🔴" if cls_name == "defect" else "🟢"
                    st.write(f"{icon} **{cls_name.upper()}** - confidence: {conf:.2%}")
            else:
                st.info("No detections found.")
        else:
            st.error(f"Model not found: {model_path}")
    else:
        st.info("Upload an image to run detection.")

st.divider()
st.subheader("Privacy Guarantee")
c1, c2, c3 = st.columns(3)
c1.info("Raw images never leave the factory. Only model weight updates are transmitted.")
c2.info("Gaussian noise is added to gradients before transmission (differential privacy).")
c3.info("Epsilon budget is tracked per round to ensure total privacy cost stays bounded.")
