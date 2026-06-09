# 🏭 Federated CV Quality Control

> Privacy-preserving industrial defect detection using Federated Learning, YOLOv8, and Differential Privacy — raw images never leave the factory.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://federated-cv-qc.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.14-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![Flower](https://img.shields.io/badge/Federated-Flower_FL-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

In industrial manufacturing, factories cannot share product images due to **data privacy regulations** and **competitive confidentiality**. Yet each factory individually has insufficient data to train a reliable defect detector.

**The challenge:** How do you train a high-quality defect detection model across multiple factories without any factory ever exposing its raw image data?

**The solution:** Federated Learning — each factory trains a local model on its own data, transmits only noise-injected weight updates to a central aggregation server, and receives an improved global model in return. No raw images are ever shared.

---

## 🎯 Project Overview

This system simulates **3 independent factories**, each responsible for a different product line, collaboratively training a shared defect detection model using the **Flower federated learning framework** and **YOLOv8 classification**.

| Factory | Product | Dataset Category |
|---|---|---|
| Factory A | Metal Nuts | MVTec AD — `metal_nut` |
| Factory B | Transistors | MVTec AD — `transistor` |
| Factory C | Carpet / Fabric | MVTec AD — `carpet` |

Each factory trains a **YOLOv8n-cls** model locally, injects **Gaussian differential privacy noise** into gradients before transmission, and the server aggregates updates using **FedAvg**. A **Streamlit dashboard** provides real-time monitoring and live inference.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FACTORY CLIENTS (Edge)                      │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Factory A   │  │  Factory B   │  │  Factory C   │          │
│  │  Metal Nuts  │  │ Transistors  │  │   Carpet     │          │
│  │              │  │              │  │              │          │
│  │ CCTV Images  │  │ CCTV Images  │  │ CCTV Images  │          │
│  │     ↓        │  │     ↓        │  │     ↓        │          │
│  │ YOLOv8-cls   │  │ YOLOv8-cls   │  │ YOLOv8-cls   │          │
│  │ Local Train  │  │ Local Train  │  │ Local Train  │          │
│  │     ↓        │  │     ↓        │  │     ↓        │          │
│  │ DP Noise +   │  │ DP Noise +   │  │ DP Noise +   │          │
│  │ Weight ΔW    │  │ Weight ΔW    │  │ Weight ΔW    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                    ΔW only (no images)                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEDERATED AGGREGATION SERVER                   │
│                                                                 │
│              Flower (flwr) — FedAvg Strategy                    │
│                                                                 │
│   ┌─────────────────────┐   ┌─────────────────────────────┐     │
│   │   FedAvg Aggregator │   │   DP Budget Tracker          │     │
│   │                     │   │                             │     │
│   │  Wᵍ ← avg(ΔW₁,      │   │  ε_total += ε_per_round     │     │
│   │         ΔW₂, ΔW₃)   │   │  Logs per round to JSON     │     │
│   └─────────────────────┘   └─────────────────────────────┘     │
│                                                                 │
│              Global model Wᵍ pushed back to all clients         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD                           │
│                                                                 │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│   │  Live        │  │  Training    │  │  DP ε Budget         │  │
│   │  Inference   │  │  Progress    │  │  Consumption Chart   │  │
│   │  per Factory │  │  mAP/Round   │  │                      │  │
│   └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔒 Privacy Architecture

```
                    What is transmitted
                    ───────────────────
Factory A  ──────►  ΔW + Gaussian noise (ε=0.5 per round)  ──────► Server
Factory B  ──────►  ΔW + Gaussian noise (ε=0.5 per round)  ──────► Server
Factory C  ──────►  ΔW + Gaussian noise (ε=0.5 per round)  ──────► Server

                    What is NEVER transmitted
                    ─────────────────────────
                    ✗ Raw product images
                    ✗ Local dataset statistics
                    ✗ Factory-specific metadata

                    Privacy budget
                    ──────────────
                    ε = 0.5 per round × 10 rounds = ε_total = 5.0
                    δ = 1e-5 (probability of privacy breach)
```

---

## 📁 Project Structure

```
federated-cv-qc/
│
├── 📂 clients/
│   ├── client.py              # Flower client — YOLO detection mode (v1)
│   └── client_cls.py          # Flower client — YOLO classification mode (v2)
│
├── 📂 server/
│   ├── server.py              # FL server — detection mode (v1)
│   ├── server_cls.py          # FL server — classification mode (v2)
│   └── logs/
│       └── training_log.json  # Per-round metrics: epsilon, mAP, clients
│
├── 📂 dashboard/
│   └── app.py                 # Streamlit monitoring dashboard
│
├── 📂 data/
│   ├── prepare_dataset.py     # MVTec → YOLO detection format converter
│   ├── prepare_cls.py         # MVTec → YOLO classification format converter
│   ├── factory_a/             # Metal nut images + dataset.yaml
│   ├── factory_b/             # Transistor images + dataset.yaml
│   └── factory_c/             # Carpet images + dataset.yaml
│
├── 📂 runs/
│   └── classify/
│       └── runs/cls/
│           ├── factory_a/train/weights/best.pt   # Trained model — Factory A
│           ├── factory_b/train/weights/best.pt   # Trained model — Factory B
│           └── factory_c/train/weights/best.pt   # Trained model — Factory C
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Model | YOLOv8n-cls (Ultralytics) | Image classification — good vs defect |
| Federated Learning | Flower (`flwr`) | Client-server FL orchestration |
| Aggregation | FedAvg | Weighted average of client updates |
| Differential Privacy | `diffprivlib` | Gaussian noise injection on gradients |
| Dashboard | Streamlit | Live inference + training monitoring |
| Dataset | MVTec Anomaly Detection | Industry-standard QC benchmark |
| Language | Python 3.14 | Core implementation |
| Deep Learning | PyTorch + Torchvision | Model backend |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Windows / Linux / Mac
- 8GB RAM minimum (no GPU required)

### 1. Clone the repository
```bash
git clone https://github.com/Atharv-725/federated-cv-qc.git
cd federated-cv-qc
```

### 2. Create virtual environment
```bash
uv venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies
```bash
uv pip install flwr ultralytics opencv-python diffprivlib fastapi uvicorn streamlit torch torchvision pillow numpy pandas matplotlib
```

### 4. Download MVTec AD Dataset
Download from: https://www.mvtec.com/company/research/datasets/mvtec-ad

Download only these 3 categories (~400 MB total):
- `metal_nut.tar.xz` → extract to `data/factory_a/`
- `transistor.tar.xz` → extract to `data/factory_b/`
- `carpet.tar.xz` → extract to `data/factory_c/`

### 5. Prepare the dataset
```bash
python data/prepare_cls.py
```

---

## 🚀 Running the System

Open **4 terminals**, activate the venv in each, then run:

**Terminal 1 — Start the FL server first:**
```bash
python server/server_cls.py
```
Wait for: `Starting Federated Classification Server on port 8080...`

**Terminal 2 — Factory A client:**
```bash
python clients/client_cls.py factory_a data/factory_a/cls
```

**Terminal 3 — Factory B client:**
```bash
python clients/client_cls.py factory_b data/factory_b/cls
```

**Terminal 4 — Factory C client:**
```bash
python clients/client_cls.py factory_c data/factory_c/cls
```

Training runs for **10 rounds** (~25 minutes on CPU). After completion:

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Training Results

| Round | Clients | ε (Cumulative) | Notes |
|---|---|---|---|
| 1 | 3 | 0.5 | Initial global model |
| 3 | 3 | 1.5 | Model stabilizing |
| 5 | 3 | 2.5 | Convergence begins |
| 10 | 3 | 5.0 | Final global model |

**Total training time:** ~26 minutes (CPU, no GPU)
**Total privacy budget:** ε = 5.0, δ = 1e-5
**Model size:** ~6.2 MB per factory

---

## 🖥️ Dashboard Features

- **Live Inference** — upload any product image, select the factory model, get instant good/defect classification with confidence score
- **Class Probability Chart** — bar chart showing confidence per class
- **DP Budget Chart** — cumulative epsilon consumption across rounds
- **Round Metrics** — clients per round, rounds completed, epsilon spent

---

## 🔬 Research Connection

This project is part of a broader federated learning research track:

| Project | Description | Status |
|---|---|---|
| [Adaptive FL + Concept Drift](https://github.com/Atharv-725/adaptive-fl-paper) | FL with drift detection for non-IID data | EasyChair Preprint #52853 |
| [FedETL Framework](https://github.com/Atharv-725/federated-etl-framework) | Privacy-preserving federated ETL pipeline | IEEE Targeting |
| **Federated CV QC** (this repo) | FL for industrial defect detection | Portfolio + IEEE target |

---

## 🧠 Known Limitations & Future Work

| Limitation | Root Cause | Fix in Production |
|---|---|---|
| Low defect recall | Class imbalance (220 good vs 20 defect per factory) | Data augmentation, SMOTE, weighted loss |
| CPU-only training | No GPU available | CUDA-enabled instance |
| Simulated federation | All clients on same machine | Deploy clients on separate edge devices |
| Fixed ε per round | No adaptive privacy budget | RDP accountant for tighter bounds |

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
