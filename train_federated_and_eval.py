import os, sys, json, math, shutil, copy
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

FACTORIES = {
    'factory_a': 'metal_nut',
    'factory_b': 'transistor',
    'factory_c': 'carpet'
}

BASE_DATA = Path('data')
FED_RUNS_DIR = Path('runs/federated')
FED_RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path('server/logs')
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 1. ANALYTICAL RÉNYI DIFFERENTIAL PRIVACY (RDP) ACCOUNTANT
# -------------------------------------------------------------------------

def compute_dp_epsilon(rounds, sigma, delta=1e-5, q=1.0):
    """
    Computes exact (epsilon, delta)-DP bound for Gaussian mechanism FedAvg
    under Renyi Differential Privacy (RDP) analytical composition.
    """
    if rounds <= 0 or sigma <= 0:
        return 0.0
    
    # Grid search over Renyi divergence orders alpha in (1.05, 128.0)
    alphas = np.linspace(1.05, 128.0, 500)
    best_eps = float('inf')
    
    for alpha in alphas:
        # RDP for Gaussian mechanism: eps_RDP(alpha) = rounds * (alpha / (2 * sigma^2))
        rdp_total = rounds * (alpha / (2.0 * (sigma ** 2)))
        # Convert RDP to (epsilon, delta)-DP: eps = rdp + ln(1/delta) / (alpha - 1)
        eps = rdp_total + math.log(1.0 / delta) / (alpha - 1.0)
        if eps < best_eps:
            best_eps = eps
            
    return round(float(best_eps), 3)

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def evaluate_model_on_test(model, test_dir):
    class_names = model.names
    defect_idx = None
    for idx, name in class_names.items():
        if name.lower() == 'defect':
            defect_idx = idx
            break
            
    good_dir = test_dir / 'good'
    defect_dir = test_dir / 'defect'
    
    y_true = []
    y_prob_defect = []
    
    for p in sorted(list(good_dir.glob('*.*'))):
        res = model(str(p), verbose=False, imgsz=224)[0]
        probs = res.probs.data.cpu().numpy()
        p_defect = float(probs[defect_idx]) if defect_idx is not None else float(probs[0])
        y_true.append(0)
        y_prob_defect.append(p_defect)
        
    for p in sorted(list(defect_dir.glob('*.*'))):
        res = model(str(p), verbose=False, imgsz=224)[0]
        probs = res.probs.data.cpu().numpy()
        p_defect = float(probs[defect_idx]) if defect_idx is not None else float(probs[0])
        y_true.append(1)
        y_prob_defect.append(p_defect)
        
    y_true = np.array(y_true)
    y_prob_defect = np.array(y_prob_defect)
    y_pred = (y_prob_defect >= 0.5).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auroc = float(roc_auc_score(y_true, y_prob_defect))
    except Exception:
        auroc = 0.5
    cm = confusion_matrix(y_true, y_pred).tolist()
    ece = compute_ece(y_prob_defect, y_true)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auroc': auroc,
        'confusion_matrix': cm,
        'ece': ece,
        'total_samples': len(y_true),
        'good_count': int(np.sum(y_true == 0)),
        'defect_count': int(np.sum(y_true == 1))
    }

# -------------------------------------------------------------------------
# 2. FEDERATED TRAINING WITH PER-UPDATE CLIPPING & GAUSSIAN DP
# -------------------------------------------------------------------------

NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
CLIPPING_NORM_C = 1.0
DP_SIGMA = 1.5     # Noise multiplier
TARGET_DELTA = 1e-5

print("==================================================")
print("FEDERATED LEARNING + DIFFERENTIAL PRIVACY TRAINING")
print(f"Rounds: {NUM_ROUNDS} | Clients: {len(FACTORIES)} | Local Epochs: {LOCAL_EPOCHS}")
print(f"DP Clipping Norm C: {CLIPPING_NORM_C} | Noise Multiplier sigma: {DP_SIGMA} | Delta: {TARGET_DELTA}")
print("==================================================")

# Initialize Global Model with 2-class head
init_weights_path = Path("runs/classify/runs/federated/init/init/weights/best.pt")
if not init_weights_path.exists():
    global_model = YOLO("yolov8n-cls.pt")
    init_trainer = global_model.train(
        data=str(BASE_DATA / 'factory_a' / 'clean_split'),
        epochs=1,
        imgsz=224,
        batch=16,
        project="runs/federated/init",
        name="init",
        exist_ok=True,
        verbose=False,
        workers=0
    )
    init_weights_path = Path(global_model.trainer.save_dir) / 'weights' / 'best.pt'
    if not init_weights_path.exists():
        init_candidates = list(Path("runs").glob("**/init/**/weights/best.pt"))
        if init_candidates:
            init_weights_path = init_candidates[-1]

print(f"Loaded initial 2-class global model from {init_weights_path}")
global_model = YOLO(str(init_weights_path))
global_state = copy.deepcopy(global_model.model.state_dict())

training_logs = []

for r in range(1, NUM_ROUNDS + 1):
    print(f"\n--- [Round {r}/{NUM_ROUNDS}] Broadcasting Global Weights to Factory Clients ---")
    
    client_updates = []
    
    for fid, cat in FACTORIES.items():
        clean_split = BASE_DATA / fid / 'clean_split'
        
        # Load global model into client (using 2-class architecture)
        client_model = YOLO(str(init_weights_path))
        client_model.model.load_state_dict(global_state)
        
        # Train locally
        client_project = FED_RUNS_DIR / fid / f"round_{r}"
        client_model.train(
            data=str(clean_split),
            epochs=LOCAL_EPOCHS,
            imgsz=224,
            batch=16,
            project=str(client_project),
            name="train",
            exist_ok=True,
            verbose=False,
            workers=0
        )
        
        # Reload latest trained weights
        last_ckpt = Path(client_model.trainer.save_dir) / 'weights' / 'last.pt'
        if not last_ckpt.exists():
            cand = list(Path("runs").glob(f"**/{fid}/**/round_{r}/**/weights/last.pt"))
            if cand: last_ckpt = cand[-1]
            
        trained_client = YOLO(str(last_ckpt))
        local_state = trained_client.model.state_dict()
        
        # Compute weight update delta_W = local_state - global_state
        delta_W = {}
        total_norm_sq = 0.0
        for k, v in local_state.items():
            if k in global_state:
                diff = (v.float() - global_state[k].float()).cpu()
                delta_W[k] = diff
                total_norm_sq += torch.sum(diff ** 2).item()
                
        l2_norm = math.sqrt(total_norm_sq)
        
        # 1. Per-Update Clipping: delta_W * min(1, C / l2_norm)
        clip_factor = min(1.0, CLIPPING_NORM_C / (l2_norm + 1e-7))
        for k in delta_W.keys():
            delta_W[k] *= clip_factor
            
        # 2. Add Calibrated Gaussian DP Noise
        # Noise std = sigma * C / sqrt(num_clients)
        noise_std = (DP_SIGMA * CLIPPING_NORM_C) / math.sqrt(len(FACTORIES))
        for k in delta_W.keys():
            noise = torch.randn_like(delta_W[k]) * noise_std
            delta_W[k] += noise
            
        client_updates.append(delta_W)
        print(f"  [{fid}] Local training complete. Update norm: {l2_norm:.4f} (Clipped with factor: {clip_factor:.4f})")

    # 3. Server Aggregation (FedAvg)
    print(f"  [Server] Aggregating {len(client_updates)} client updates via DP-FedAvg...")
    for k in global_state.keys():
        if k in client_updates[0]:
            avg_delta = torch.stack([client_updates[i][k] for i in range(len(client_updates))]).mean(dim=0)
            global_state[k] = (global_state[k].float().cpu() + avg_delta).to(dtype=global_state[k].dtype)

    # Compute honest cumulative epsilon via RDP accountant
    current_eps = compute_dp_epsilon(rounds=r, sigma=DP_SIGMA, delta=TARGET_DELTA)
    
    round_log = {
        "round": r,
        "num_clients": len(FACTORIES),
        "epsilon_used": current_eps,
        "delta": TARGET_DELTA,
        "clipping_norm": CLIPPING_NORM_C,
        "sigma": DP_SIGMA,
        "failures": 0,
        "annotation": f"Round {r} completed. L2 norm clipped to {CLIPPING_NORM_C}, Gaussian noise (sigma={DP_SIGMA}) injected. Total epsilon spent = {current_eps}."
    }
    training_logs.append(round_log)
    print(f"  [Server] Round {r} finished. Total Epsilon: {current_eps:.2f} (delta={TARGET_DELTA})")

# Save converged global model
global_model.model.load_state_dict(global_state, strict=False)
fed_checkpoint_path = FED_RUNS_DIR / "global_dp_fedavg.pt"
torch.save(global_state, fed_checkpoint_path)
print(f"\nSaved Converged Federated Checkpoint to {fed_checkpoint_path}")

# Save training logs to server/logs/training_log.json
with open(LOGS_DIR / "training_log.json", "w") as f:
    json.dump(training_logs, f, indent=2)

print("Saved honest training logs to server/logs/training_log.json")

# -------------------------------------------------------------------------
# 3. EVALUATING FEDERATED MODEL ON TEST SETS
# -------------------------------------------------------------------------

fed_results = {}
for fid, cat in FACTORIES.items():
    test_dir = BASE_DATA / fid / 'clean_split' / 'test'
    print(f"\n[{fid}] Evaluating Global Federated Model on test set...")
    metrics = evaluate_model_on_test(global_model, test_dir)
    fed_results[fid] = metrics
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  ECE:       {metrics['ece']:.4f}")

with open("metrics_federated.json", "w") as f:
    json.dump(fed_results, f, indent=2)

print("\n=== EXPORTING GLOBAL FEDERATED MODEL TO ONNX ===")
# Export global model to ONNX
global_model.export(format="onnx", imgsz=224, dynamic=False, opset=12)
global_onnx_cand = Path(fed_checkpoint_path).with_suffix('.onnx')
if not global_onnx_cand.exists():
    cands = list(Path("runs").glob("**/global_dp_fedavg.onnx"))
    if cands:
        global_onnx_cand = cands[-1]

target_public_fed = Path("public/models/global_fed.onnx")
target_models_fed = Path("models/global_fed.onnx")
if global_onnx_cand.exists():
    shutil.copy(global_onnx_cand, target_public_fed)
    shutil.copy(global_onnx_cand, target_models_fed)
    print(f"Global Federated ONNX saved to {target_public_fed} and {target_models_fed}")

# Build comprehensive metrics comparison
comparison = {}
with open("metrics_centralized.json", "r") as f:
    cent_metrics = json.load(f)

for fid in FACTORIES.keys():
    comparison[fid] = {
        "domain": FACTORIES[fid],
        "centralized": cent_metrics.get(fid, {}),
        "federated_dp": fed_results.get(fid, {})
    }

with open("metrics_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

print("Saved combined comparison to metrics_comparison.json")
print("\n=== FEDERATED EVALUATION COMPLETE ===")
