import os, shutil, random
from pathlib import Path

FACTORIES = {
    "factory_a": "metal_nut",
    "factory_b": "transistor",
    "factory_c": "carpet",
}

BASE = Path("data")

def create_yolo_label(label_path, class_id):
    label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")

def process_factory(factory, category):
    src = BASE / factory / category
    out = BASE / factory / "yolo"

    for split in ["train", "val"]:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    for img in (src / "train" / "good").glob("*.png"):
        dst_img = out / "train" / "images" / f"good_{img.name}"
        dst_lbl = out / "train" / "labels" / f"good_{img.stem}.txt"
        shutil.copy(img, dst_img)
        create_yolo_label(dst_lbl, 0)

    for img in (src / "test" / "good").glob("*.png"):
        dst_img = out / "val" / "images" / f"good_{img.name}"
        dst_lbl = out / "val" / "labels" / f"good_{img.stem}.txt"
        shutil.copy(img, dst_img)
        create_yolo_label(dst_lbl, 0)

    for defect_dir in (src / "test").iterdir():
        if defect_dir.is_dir() and defect_dir.name != "good":
            for img in defect_dir.glob("*.png"):
                unique_name = f"defect_{defect_dir.name}_{img.name}"
                dst_img = out / "val" / "images" / unique_name
                dst_lbl = out / "val" / "labels" / f"defect_{defect_dir.name}_{img.stem}.txt"
                shutil.copy(img, dst_img)
                create_yolo_label(dst_lbl, 1)

    abs_out = out.resolve().as_posix()
    yaml_path = BASE / factory / "dataset.yaml"
    yaml_path.write_text(
        f"path: {abs_out}\ntrain: train/images\nval: val/images\nnc: 2\nnames: ['good', 'defect']\n"
    )

    train_count = len(list((out / "train" / "images").glob("*.png")))
    val_count   = len(list((out / "val" / "images").glob("*.png")))
    print(f"[{factory}] {category}: {train_count} train, {val_count} val -> {yaml_path}")

if __name__ == "__main__":
    random.seed(42)
    for factory, category in FACTORIES.items():
        src_check = BASE / factory / category
        if not src_check.exists():
            print(f"SKIP {factory}: {src_check} not found")
            continue
        process_factory(factory, category)
    print("Done.")
