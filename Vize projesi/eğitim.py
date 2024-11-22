from roboflow import Roboflow

import torch
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli


rf = Roboflow(api_key="TM3T4364h8X4PSk2THPv")
project = rf.workspace("yusuf-ll5dq").project("vize-yolo")
version = project.version(1)
dataset = version.download("yolov8")

# Modeli eğit
if __name__ == "__main__" :
    model.train(data=f"{dataset.location}/data.yaml", epochs=50)
