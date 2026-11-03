from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import torch


TEST_IMAGES = Path("archive/RDD_SPLIT/test/images")
TEST_LABELS = Path("archive/RDD_SPLIT/test/labels")
MODEL_PATH = Path("Models/best_fixed.pt")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class_names = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "other corruption",
    "pothole"
]


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not TEST_IMAGES.exists() or not TEST_LABELS.exists():
        raise FileNotFoundError("Test dataset not found.")

    images = [
        p for p in TEST_IMAGES.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(TEST_LABELS.glob("*.txt"))

    print("Test images:", len(images))
    print("Label files:", len(labels))

    class_counts = Counter()

    for file in labels:
        with open(file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1

    print("\nGround-truth class distribution:")

    for i, name in enumerate(class_names):
        print(f"{name}: {class_counts[i]}")

    device = 0 if torch.cuda.is_available() else "cpu"

    if device == 0:
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data="data.yaml",
        split="test",
        imgsz=416,
        batch=1,
        device=device,
        workers=0,
        conf=0.1,
        plots=True,
        project="runs/evaluation",
        name="RDD2022_test_results"
    )

    p = metrics.box.mp
    r = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    f1 = 2 * p * r / (p + r) if p + r > 0 else 0

    print("\n========== OVERALL PERFORMANCE ==========")
    print(f"Precision: {p:.4f} ({p*100:.2f}%)")
    print(f"Recall: {r:.4f} ({r*100:.2f}%)")
    print(f"F1 Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"mAP50: {map50:.4f} ({map50*100:.2f}%)")
    print(f"mAP50-95: {map5095:.4f} ({map5095*100:.2f}%)")

    print("\n========== CLASS-WISE PERFORMANCE ==========")

    for i, name in enumerate(class_names):

        cp, cr, ap50, ap = metrics.box.class_result(i)

        cf1 = 2 * cp * cr / (cp + cr) if cp + cr > 0 else 0

        print(f"\nClass: {name}")
        print(f"Precision: {cp:.4f}")
        print(f"Recall: {cr:.4f}")
        print(f"F1 Score: {cf1:.4f}")
        print(f"mAP50: {ap50:.4f}")
        print(f"mAP50-95: {ap:.4f}")

    print("\nEvaluation completed successfully!")
    print("Results saved in runs/evaluation/")


if __name__ == "__main__":
    main()