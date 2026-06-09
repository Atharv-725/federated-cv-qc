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
    col1, col2, col3 = st.columns(3)
    col1.metric("Rounds Completed", int(df["round"].max()))
    col2.metric("Total Epsilon Spent", f"{df['epsilon_used'].max():.2f}")
    col3.metric("Clients per Round", int(df["num_clients"].iloc[-1]))
    st.subheader("Training Progress")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["round"], df["epsilon_used"], color="#534AB7", marker="o", linewidth=2)
    ax.set_title("DP Epsilon Budget Consumed per Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Epsilon")
    ax.grid(True, alpha=0.3)
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
        st.image(img, caption="Input image", width=300)
        model_path = os.path.join(BASE_DIR, "..", "runs", "classify", "runs", "cls", factory, "train", "weights", "best.pt")
        if os.path.exists(model_path):
            model = YOLO(model_path)
            results = model(img, verbose=False)
            probs = results[0].probs
            top1_idx = probs.top1
            top1_label = model.names[top1_idx]
            top1_conf = float(probs.top1conf)
            all_probs = probs.data.tolist()
            if top1_label == "defect":
                st.error(f"DEFECT DETECTED — confidence: {top1_conf:.2%}")
            else:
                st.success(f"GOOD PRODUCT — confidence: {top1_conf:.2%}")
            st.subheader("Class Probabilities")
            prob_df = pd.DataFrame({
                "Class": [model.names[i] for i in range(len(all_probs))],
                "Confidence": [round(p, 4) for p in all_probs]
            })
            st.bar_chart(prob_df.set_index("Class"))
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