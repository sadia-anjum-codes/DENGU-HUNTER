from pathlib import Path
from ultralytics import YOLO
import yaml

root = Path(r"D:\DENGUE_HUNTER")
candidates = list(root.rglob("valid/images"))

if not candidates:
    print("ERROR: valid/images folder not found")
    raise SystemExit(1)

valid = candidates[0]
dataset_root = valid.parent.parent
print("DATASET ROOT:", dataset_root)

config = {
    "path": str(dataset_root),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 5,
    "names": ["Bottle", "Coconut-Exocarp", "Drain-Inlet", "Tire", "Vase"]
}

for split in ("train", "valid", "test"):
    folder = dataset_root / split / "images"
    print(split, "exists:", folder.is_dir())

yaml_path = Path("evaluation_fixed.yaml").resolve()
yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

model = YOLO("models/f2_best.pt")
metrics = model.val(
    data=str(yaml_path),
    split="val",
    imgsz=640,
    batch=8,
    device="cpu",
    plots=True,
    project="evaluation",
    name="f2_validation",
    exist_ok=True
)

print("\n=== REAL MODEL METRICS ===")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
