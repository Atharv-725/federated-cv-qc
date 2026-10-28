from ultralytics import YOLO
from pathlib import Path
import sys

factory = sys.argv[1] if len(sys.argv) > 1 else "factory_a"

# Try candidate weight paths
candidates = [
    Path(f"runs/classify/runs/classify/runs/cls/{factory}/train/weights/best.pt"),
    Path(f"runs/classify/runs/cls/{factory}/train/weights/best.pt"),
    Path(f"runs/cls/{factory}/train/weights/best.pt"),
]

model_path = next((p for p in candidates if p.exists()), None)
if not model_path:
    print(f"Error: Model weights not found for {factory}")
    sys.exit(1)

model = YOLO(model_path)
print(f"=== Model: {model_path} ===")
print("Class names:", model.names)

val_dir = Path(f"data/{factory}/cls_balanced/val")
if not val_dir.exists():
    val_dir = Path(f"data/{factory}/cls/val")

defect_imgs = list((val_dir / "defect").glob("*.png"))[:4]
good_imgs   = list((val_dir / "good").glob("*.png"))[:4]

print("\n--- Defect Validation Samples ---")
for img in defect_imgs:
    r = model(img, verbose=False)
    probs = r[0].probs
    top1 = probs.top1
    conf = float(probs.top1conf)
    print(f"  {img.name[:25]:<26} -> Predicted: {model.names[top1]:<8} | Confidence: {conf:.1%}")

print("\n--- Good / Nominal Validation Samples ---")
for img in good_imgs:
    r = model(img, verbose=False)
    probs = r[0].probs
    top1 = probs.top1
    conf = float(probs.top1conf)
    print(f"  {img.name[:25]:<26} -> Predicted: {model.names[top1]:<8} | Confidence: {conf:.1%}")