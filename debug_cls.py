from ultralytics import YOLO
from pathlib import Path

model_path = "runs/classify/runs/cls/factory_a/train/weights/best.pt"
model = YOLO(model_path)
print("Class names:", model.names)

test_dir = Path("data/factory_a/cls/val")
defect_imgs = list((test_dir / "defect").glob("*.png"))[:3]
good_imgs   = list((test_dir / "good").glob("*.png"))[:3]

print("\n--- Defect images ---")
for img in defect_imgs:
    r = model(img, verbose=False)
    probs = r[0].probs
    print(f"{img.name}: top1={probs.top1} ({model.names[probs.top1]}) conf={probs.top1conf:.3f}")

print("\n--- Good images ---")
for img in good_imgs:
    r = model(img, verbose=False)
    probs = r[0].probs
    print(f"{img.name}: top1={probs.top1} ({model.names[probs.top1]}) conf={probs.top1conf:.3f}")