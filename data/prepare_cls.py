import os, shutil
from pathlib import Path

FACTORIES = {
    "factory_a": "metal_nut",
    "factory_b": "transistor",
    "factory_c": "carpet",
}

BASE = Path("data")

for factory, category in FACTORIES.items():
    src = BASE / factory / category
    out = BASE / factory / "cls"

    for split in ["train", "val"]:
        (out / split / "good").mkdir(parents=True, exist_ok=True)
        (out / split / "defect").mkdir(parents=True, exist_ok=True)

    # Train — only good images
    for img in (src / "train" / "good").glob("*.png"):
        shutil.copy(img, out / "train" / "good" / img.name)

    # Val — good
    for img in (src / "test" / "good").glob("*.png"):
        shutil.copy(img, out / "val" / "good" / img.name)

    # Val — defect (all defect subfolders)
    for defect_dir in (src / "test").iterdir():
        if defect_dir.is_dir() and defect_dir.name != "good":
            for img in defect_dir.glob("*.png"):
                unique = f"{defect_dir.name}_{img.name}"
                shutil.copy(img, out / "val" / "defect" / unique)

    # Also add some defect images to train by copying from val
    for defect_dir in (src / "test").iterdir():
        if defect_dir.is_dir() and defect_dir.name != "good":
            for img in list(defect_dir.glob("*.png"))[:5]:
                unique = f"{defect_dir.name}_{img.name}"
                shutil.copy(img, out / "train" / "defect" / unique)

    train_good = len(list((out / "train" / "good").glob("*.png")))
    train_def  = len(list((out / "train" / "defect").glob("*.png")))
    val_good   = len(list((out / "val" / "good").glob("*.png")))
    val_def    = len(list((out / "val" / "defect").glob("*.png")))
    print(f"[{factory}] train: {train_good} good, {train_def} defect | val: {val_good} good, {val_def} defect")

print("Done.")