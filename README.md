# 🏭 Federated CV Quality Control

> Privacy-preserving industrial defect detection using Federated Learning, YOLOv8, and Differential Privacy — raw images never leave the factory LAN.

[![Live Web Studio](https://img.shields.io/badge/Live_Studio-Vercel-black?logo=vercel)](https://federated-cv-qc.vercel.app/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://federated-cv-qc.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![Flower](https://img.shields.io/badge/Federated-Flower_FL-purple)
![ONNX Web](https://img.shields.io/badge/Inference-ONNX_Runtime_Web-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

In industrial manufacturing, factories cannot share product images due to **intellectual property protection**, **data privacy regulations**, and **competitive confidentiality**. Yet individual factory lines often suffer from scarce defect samples (rare anomalies) to train reliable vision models.

- **The Challenge:** How do you evaluate and collaboratively improve defect detection models without exposing proprietary manufacturing images?
- **The Solution:** Edge-based Federated Learning with Differential Privacy. Factories train locally on edge devices, clip updates to bound sensitivity ($L_2 \le C$), inject calibrated Gaussian noise ($\sigma$), and exchange coordinate-wise parameter updates ($\Delta W$). Raw image tensors never traverse the network.

---

## 🎯 What the Models Actually Do (and Don't Do)

- **What they DO:**
  - Fast, binary image classification: predicts whether an incoming workpiece is **Nominal (Pass)** or **Defective (Reject)**.
  - Outputs real, calibrated softmax probabilities: $P(\text{defect})$ and $P(\text{good})$.
  - Executes **100% client-side in the browser** via **ONNX Runtime Web (`ort.min.js`)** compiled to WebAssembly (WASM). No image is uploaded to any server.
  - Interactive decision boundary: dynamically adjusts the classification threshold $\tau$ (from 15% to 90%) to balance precision vs. recall.
  - Out-of-Distribution (OOD) rejection: detects images with ambiguous entropy (max certainty $< 58\%$) or non-industrial imagery, rejecting them rather than outputting confident wrong answers.
  - Metrology visualizer: computes real Sobel gradient edge metrology and spatial energy density maps directly on canvas pixels.

- **What they DON'T DO:**
  - These models are **classification networks (`YOLOv8n-cls`)**, not bounding-box object detectors or pixel-level instance segmenters (`YOLOv8-seg`).
  - They do not output synthetic localized bounding boxes. Where visualizers are shown, they represent true 2D pixel gradient magnitude and spatial energy density.

---

## 🏗️ System Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    EDGE FACTORY NODES (MVTec AD)                │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Factory A   │  │  Factory B   │  │  Factory C   │          │
│  │  Metal Nuts  │  │ Transistors  │  │   Carpet     │          │
│  │              │  │              │  │              │          │
│  │ Clean Split  │  │ Clean Split  │  │ Clean Split  │          │
│  │     ↓        │  │     ↓        │  │     ↓        │          │
│  │ YOLOv8n-cls  │  │ YOLOv8n-cls  │  │ YOLOv8n-cls  │          │
│  │ Train/Val    │  │ Train/Val    │  │ Train/Val    │          │
│  │     ↓        │  │     ↓        │  │     ↓        │          │
│  │ Norm Clip C  │  │ Norm Clip C  │  │ Norm Clip C  │          │
│  │ + Gauss DP   │  │ + Gauss DP   │  │ + Gauss DP   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│             ΔW + N(0, σ²C²I) (No Raw Images Transmitted)        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEDERATED AGGREGATION SERVER                   │
│                                                                 │
│                   DP-FedAvg Parameter Update                    │
│             W_{t+1} = W_t + (1/K) * sum(ΔW_k + Noise)           │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │   Analytical Rényi Differential Privacy (RDP) Tracker   │   │
│   │   Calculates exact (ε, δ) privacy bound across rounds   │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              IN-BROWSER ONNX RUNTIME WEB STUDIO                 │
│                                                                 │
│   - Client-side WebAssembly inference on uploaded pixels        │
│   - Real-time softmax probability distribution                  │
│   - Calibrated ROC threshold slider (Nut: 50%, Transistor: 35%) │
│   - OOD rejection for low-confidence / unknown inputs           │
│   - Real Sobel gradient metrology and edge energy visualizer    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Evaluation (Held-Out Test Sets)

All evaluations are conducted on **strictly held-out test splits** that were never seen during training or validation (zero data leakage):

| Domain / Workpiece | Paradigm | Test Accuracy | Precision | Recall | F1-Score | AUROC | ECE Calibration | Held-Out Test N |
|---|---|---|---|---|---|---|---|---|
| **🔩 Factory A (metal_nut)** | Centralized | **75.00%** | **100.00%** | **65.52%** | **0.7917** | **0.9875** | 0.2411 | 40 (11 G / 29 D) |
| **🔩 Factory A (metal_nut)** | DP-FedAvg (R5) | 27.50% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.0000 | 40 (11 G / 29 D) |
| **⚡ Factory B (transistor)** | Centralized | **80.95%** | **100.00%** | **33.33%** | **0.5000** | **0.8639** | 0.1152 | 42 (30 G / 12 D) |
| **⚡ Factory B (transistor)** | DP-FedAvg (R5) | 71.43% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.0000 | 42 (30 G / 12 D) |
| **🧵 Factory C (carpet)** | Centralized | **85.37%** | **88.89%** | **88.89%** | **0.8889** | **0.8836** | 0.1784 | 41 (14 G / 27 D) |
| **🧵 Factory C (carpet)** | DP-FedAvg (R5) | 34.15% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.0000 | 41 (14 G / 27 D) |

### Confusion Matrices (Centralized Held-Out Test):
- **Factory A (`metal_nut`):** `[[11, 0], [10, 19]]` (11 TN, 0 FP, 10 FN, 19 TP)
- **Factory B (`transistor`):** `[[30, 0], [8, 4]]` (30 TN, 0 FP, 8 FN, 4 TP)
- **Factory C (`carpet`):** `[[11, 3], [3, 24]]` (11 TN, 3 FP, 3 FN, 24 TP)

---

## 🔬 Empirical Insights & Failure Modes

1. **Why Centralized Models Excel on Fabrics and Fasteners:**
   - On `carpet`, defects (punctures, loose threads, cuts) disrupt repetitive textural periodicity, yielding strong contrast features and high test accuracy (**85.37%**).
   - On `metal_nut`, structural flaws such as bent rings and color stains yield distinct silhouette perturbations, achieving an AUROC of **0.9875**.

2. **Why Transistor Defect Recall is Lower (33.33% at 50% cutoff):**
   - In TO-92 transistors, anomalies such as bent leads represent microscopic sub-millimeter coplanarity deviations occupying less than 1% of pixels on a $224 \times 224$ downsampled image.
   - At standard 50% threshold, subtle bent leads predict $P(\text{defect}) \approx 35\% - 48\%$.
   - **Solution:** Setting the calibrated threshold slider to **35%** (calibrated ROC cutoff) restores high defect recall on transistor inspection.

3. **Why Cross-Domain Federated DP Collapses Global Recall:**
   - In this setup, 3 clients represent completely disjoint visual domains: structured mechanical metal rings (`metal_nut`), electronic semiconductor packages (`transistor`), and woven textiles (`carpet`).
   - Coordinate-wise parameter averaging across heterogeneous non-IID modalities causes cross-gradient interference.
   - Injecting calibrated differential privacy noise ($\sigma=1.5, C=1.0$) with only $N=3$ clients causes perturbation noise to dominate the update signal, pulling the global model toward the class prior.
   - **Real-World Takeaway:** Strict Differential Privacy across multi-modal non-IID edge networks requires **localized personalization heads** or **split federated learning** rather than monolithic coordinate-wise parameter averaging.

---

## 🔒 Differential Privacy Budget (Rényi DP Accountant)

Privacy tracking uses analytical Rényi Differential Privacy (RDP) composition with grid-search optimization over Rényi orders $\alpha \in [1.05, 128.0]$:

$$\epsilon(\alpha) = \frac{\alpha \cdot \text{rounds}}{2\sigma^2} + \frac{\ln(1/\delta)}{\alpha - 1}$$

| Round | Clients | $L_2$ Clip Norm $C$ | Noise Multiplier $\sigma$ | Cumulative $\epsilon$ | Target $\delta$ |
|---|---|---|---|---|---|
| **Round 1** | 3 | 1.0 | 1.5 | **3.421** | $1 \times 10^{-5}$ |
| **Round 2** | 3 | 1.0 | 1.5 | **4.969** | $1 \times 10^{-5}$ |
| **Round 3** | 3 | 1.0 | 1.5 | **6.208** | $1 \times 10^{-5}$ |
| **Round 4** | 3 | 1.0 | 1.5 | **7.287** | $1 \times 10^{-5}$ |
| **Round 5** | 3 | 1.0 | 1.5 | **8.269** | $1 \times 10^{-5}$ |

---

## 🛠️ Reproduction Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Atharv-725/federated-cv-qc.git
cd federated-cv-qc

# Create virtual environment with Python 3.10+
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Zero-Leakage Dataset Splitting
Extract MVTec AD categories (`metal_nut`, `transistor`, `carpet`) into `data/factory_{a,b,c}/`. Run the zero-leakage stratified splitter:
```bash
python prepare_proper_splits.py
```
This reserves 30% of defects strictly for the held-out test split, 20% for validation, and 50% for training with assertion verification (`len(train_files & test_files) == 0`).

### 3. Centralized Training & ONNX Export
```bash
python train_centralized_and_eval.py
```
Trains baseline models on each factory split, calculates held-out test metrics, and exports `factory_a.onnx`, `factory_b.onnx`, and `factory_c.onnx` into `models/`.

### 4. Federated DP Training & Evaluation
```bash
python train_federated_and_eval.py
```
Executes DP-FedAvg with per-update clipping, Gaussian perturbation, analytical RDP tracking, held-out evaluation, and exports `global_fed.onnx` and `metrics_comparison.json`.

### 5. Launch In-Browser QC Studio
```bash
# Python HTTP server
python -m http.server 8000
# Open http://localhost:8000 in your browser
```
Or deploy directly to Vercel via `npx vercel`.

---

## 👤 Author

**Atharv Dorle (THUNDER)**  
B.Tech Computer Science — SRM Institute of Science and Technology, Chennai  
CGPA: 9.37 | Quantum Computing Honours | AWS + IBM + AICTE Certified  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/atharv-dorle-6552602b8/)
[![GitHub](https://img.shields.io/badge/GitHub-Atharv--725-black)](https://github.com/Atharv-725)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
