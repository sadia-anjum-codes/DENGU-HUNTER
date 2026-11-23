from pathlib import Path
from ultralytics import YOLO
import yaml

root = Path(r"D:\DENGUE_HUNTER\Breeding Place Detection\Breeding Place Detection")
original = root / "data.yaml"

with open(original, encoding="utf-8") as f:
    config = yaml.safe_load(f)

for split in ("train", "val", "test"):
    if split in config:
        folder = (root / config[split]).resolve()
        config[split] = str(folder)
        print(f"{split}: {folder} | exists={folder.is_dir()}")

evaluation_yaml = root / "evaluation_data.yaml"

with open(evaluation_yaml, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)

model = YOLO("models/f2_best.pt")

metrics = model.val(
    data=str(evaluation_yaml),
    split="val",
    imgsz=640,
    batch=8,
    device="cpu",
    plots=True,
    project="evaluation",
    name="f2_validation",
    exist_ok=True
)

print("\n========== REAL MODEL METRICS ==========")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")
print("========================================")
