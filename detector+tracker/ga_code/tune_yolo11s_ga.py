# coding=utf-8
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.tune(
    data="datasets/AntiUAV300_YOLO_VISIBLE/anti_uav.yaml",
    epochs=10,
    iterations=10,
    imgsz=640,
    batch=16,
    device=0,
    workers=2,
    optimizer="auto",
    plots=True,
    save=True,
    val=True
)