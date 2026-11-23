from pathlib import Path
from ultralytics import YOLO
import yaml

base = Path(r"D:\DENGUE_HUNTER\Breeding Place Detection")
matches = list(base.rglob("valid/images"))

if not matches:
    print("ERROR: Breeding Place Detection validation images not found")
    raise SystemExit(1)

root = matches[0].parent.parent
print("CORRECT DATASET:", root)

valid_images = root / "valid" / "images"
valid_labels = root / "valid" / "labels"

print("Validation images:", len(list(valid_images.glob("*.*"))))
print("Validation labels:", len(list(valid_labels.glob("*.txt"))))

if not valid_labels.is_dir():
    print("ERROR: Validation labels missing")
    raise SystemExit(1)

config = {
    "path": str(root),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 5,
    "names": [
        "Bottle",
        "Coconut-Exocarp",
        "Drain-Inlet",
        "Tire",
        "Vase"
    ]
}

yaml_file = Path("evaluation_correct.yaml").resolve()
yaml_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

model = YOLO("models/f2_best.pt")

metrics = model.val(
    data=str(yaml_file),
    split="val",
    imgsz=640,
    batch=8,
    device="cpu",
    plots=True,
    project="evaluation",
    name="f2_correct_validation",
    exist_ok=True
)

print("\n=== CORRECT MODEL EVALUATION ===")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
