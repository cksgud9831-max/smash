# coding=utf-8
from ultralytics import YOLO


def main():
    model = YOLO("yolo11s.pt")

    model.train(
        data="datasets/AntiUAV300_YOLO_VISIBLE/anti_uav.yaml",

        # 팀 공통 큰 설정
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        seed=42,
        deterministic=True,
        pretrained=True,

        # GA 결과를 반영하기 위해 optimizer 고정
        optimizer="SGD",

        # GA 탐색 결과
        lr0=0.00716,
        lrf=0.01346,
        momentum=0.92767,
        weight_decay=0.00024,
        warmup_epochs=2.42352,
        warmup_momentum=0.89015,

        box=7.25046,
        cls=0.55889,
        dfl=3.26329,

        hsv_h=0.01559,
        hsv_s=0.66221,
        hsv_v=0.47878,
        degrees=0.00133,
        translate=0.13196,
        scale=0.3885,
        shear=0.00411,
        perspective=0.00007,
        flipud=0.0019,
        fliplr=0.47448,
        bgr=0.00068,
        mosaic=0.69954,
        mixup=0.00529,
        cutmix=0.00101,
        copy_paste=0.00252,
        close_mosaic=10,

        project="runs",
        name="yolo11s_ga_final"
    )


if __name__ == "__main__":
    main()