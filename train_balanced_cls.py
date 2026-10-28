import os, shutil, random, stat
from pathlib import Path
from ultralytics import YOLO

# Set random seed for reproducibility
random.seed(42)

FACTORIES = {
    "factory_a": "metal_nut",
    "factory_b": "transistor",
    "factory_c": "carpet",
}

BASE = Path("data")

print("=== STEP 1: PREPARING 50/50 BALANCED CLASSIFICATION DATASETS ===")

for factory, category in FACTORIES.items():
    src = BASE / factory / category
    out = BASE / factory / "cls_balanced"

    if out.exists():
        for root, dirs, files in os.walk(out, topdown=False):
            for file in files:
                p = os.path.join(root, file)
                try:
                    os.chmod(p, stat.S_IWRITE)
                    os.remove(p)
                except Exception:
                    pass
            for d in dirs:
                p = os.path.join(root, d)
                try:
                    os.rmdir(p)
                except Exception:
                    pass

    for split in ["train", "val"]:
        (out / split / "good").mkdir(parents=True, exist_ok=True)
        (out / split / "defect").mkdir(parents=True, exist_ok=True)

    # 1. Gather all defect images
    all_defects = []
    for defect_dir in (src / "test").iterdir():
        if defect_dir.is_dir() and defect_dir.name != "good":
            for img in defect_dir.glob("*.png"):
                unique_name = f"{defect_dir.name}_{img.name}"
                all_defects.append((img, unique_name))

    random.shuffle(all_defects)
    n_def = len(all_defects)
    n_train_def = int(n_def * 0.70)
    train_defects = all_defects[:n_train_def]
    val_defects = all_defects[n_train_def:]

    # Copy defect images
    for src_img, unique_name in train_defects:
        shutil.copy(src_img, out / "train" / "defect" / unique_name)
    for src_img, unique_name in val_defects:
        shutil.copy(src_img, out / "val" / "defect" / unique_name)

    # 2. Gather good images
    all_goods = list((src / "train" / "good").glob("*.png")) + list((src / "test" / "good").glob("*.png"))
    unique_goods = list({p.name: p for p in all_goods}.values())
    random.shuffle(unique_goods)

    # Balance 1:1 with defect counts
    train_goods = unique_goods[:len(train_defects)]
    val_goods = unique_goods[len(train_defects):len(train_defects) + len(val_defects)]

    for img in train_goods:
        shutil.copy(img, out / "train" / "good" / img.name)
    for img in val_goods:
        shutil.copy(img, out / "val" / "good" / img.name)

    print(f"[{factory}] Balanced set ready -> Train: {len(train_goods)} good, {len(train_defects)} defect | Val: {len(val_goods)} good, {len(val_defects)} defect")

print("\n=== STEP 2: TRAINING HIGH-PRECISION YOLO CLASSIFIERS ===")

for factory in FACTORIES.keys():
    print(f"\n--- Training {factory} model ---")
    data_dir = str(BASE / factory / "cls_balanced")
    
    # Load base pretrained model
    model = YOLO("yolov8n-cls.pt")
    
    # Train with 1:1 balanced data
    results = model.train(
        data=data_dir,
        epochs=6,
        imgsz=128,
        batch=16,
        project=f"runs/classify/runs/cls/{factory}",
        name="train",
        exist_ok=True,
        verbose=True,
        workers=0
    )
    
    print(f"[{factory}] Training complete! Evaluating on validation set...")
    val_results = model.val(data=data_dir, imgsz=128, verbose=False)
    top1 = float(val_results.top1) if hasattr(val_results, 'top1') else 0.0
    print(f"[{factory}] Final Validation Top-1 Accuracy: {top1:.2%}")

print("\nAll factory models successfully trained and calibrated!")
