import os, json, shutil
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

FACTORIES = {
    'factory_a': 'metal_nut',
    'factory_b': 'transistor',
    'factory_c': 'carpet'
}

BASE_DATA = Path('data')
RUNS_DIR = Path('runs/centralized')
RUNS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path('public/models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

def evaluate_on_test_set(model, test_dir):
    y_true = []
    y_prob_defect = []
    
    # Class mapping: directory names in test_dir
    # Note: Ultralytics orders classes alphabetically: 'defect': 0, 'good': 1
    # Let's inspect model.names
    class_names = model.names
    defect_idx = None
    for idx, name in class_names.items():
        if name.lower() == 'defect':
            defect_idx = idx
            break
            
    good_dir = test_dir / 'good'
    defect_dir = test_dir / 'defect'
    
    # Evaluate good images (label = 0)
    for p in sorted(list(good_dir.glob('*.*'))):
        res = model(str(p), verbose=False, imgsz=224)[0]
        probs = res.probs.data.cpu().numpy()
        p_defect = float(probs[defect_idx]) if defect_idx is not None else float(probs[0])
        y_true.append(0)
        y_prob_defect.append(p_defect)
        
    # Evaluate defect images (label = 1)
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

results = {}

for fid, cat in FACTORIES.items():
    print(f"\n==================================================")
    print(f"TRAINING CENTRALIZED BASELINE: {fid} ({cat})")
    print(f"==================================================")
    
    clean_split = BASE_DATA / fid / "clean_split"
    
    # Train YOLOv8n-cls
    model = YOLO("yolov8n-cls.pt")
    
    # Set augmentation parameters tailored for industrial QC
    # Flips and small rotations preserve structural defect traits
    is_texture = (cat == 'carpet')
    train_args = {
        'data': str(clean_split),
        'epochs': 10,
        'imgsz': 224,
        'batch': 16,
        'project': str(RUNS_DIR / fid),
        'name': 'train',
        'exist_ok': True,
        'verbose': False,
        'workers': 0,
        'degrees': 10.0,
        'fliplr': 0.5,
        'flipud': 0.5 if is_texture else 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.2,
        'hsv_v': 0.2,
        'seed': 42
    }
    
    model.train(**train_args)
    
    save_dir = Path(model.trainer.save_dir) if hasattr(model, 'trainer') and model.trainer else (RUNS_DIR / fid / 'train')
    best_weights = save_dir / 'weights' / 'best.pt'
    if not best_weights.exists():
        candidates = list(Path('runs').glob(f"**/{fid}/**/weights/best.pt"))
        if candidates:
            best_weights = candidates[-1]

    print(f"[{fid}] Loading best checkpoint from {best_weights}...")
    trained_model = YOLO(str(best_weights))
    
    # Export to ONNX for browser client-side execution
    onnx_path = MODELS_DIR / f"{fid}.onnx"
    print(f"[{fid}] Exporting best checkpoint to ONNX: {onnx_path}...")
    try:
        exported_file = trained_model.export(format="onnx", imgsz=224, dynamic=False, opset=12)
        if Path(exported_file).exists():
            shutil.copy2(exported_file, onnx_path)
            print(f"[{fid}] ONNX exported successfully: {onnx_path}")
    except Exception as e:
        print(f"[{fid}] ONNX export note: {e}")

    # Evaluate on the true, held-out test set
    test_dir = clean_split / 'test'
    print(f"[{fid}] Evaluating on strictly held-out test set ({test_dir})...")
    metrics = evaluate_on_test_set(trained_model, test_dir)
    results[fid] = metrics
    
    print(f"[{fid}] Test Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  ECE:       {metrics['ece']:.4f}")
    print(f"  Confusion Matrix: {metrics['confusion_matrix']} (TN, FP / FN, TP)")

# Save centralized metrics to json
with open("metrics_centralized.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== CENTRALIZED BASELINE EVALUATION COMPLETE ===")
print("Saved metrics to metrics_centralized.json")
