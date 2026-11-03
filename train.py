
from ultralytics import YOLO
import torch

def main():
    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=512,
        batch=2,
        device=0,
        workers=2,
        amp=True,
        cache=False,
        patience=10,
        project="runs/detect",
        name="RDD2022_YOLOv8",
        save=True,
        plots=True
    )

    print("Training completed successfully!")

if __name__ == "__main__":
    main()
