from ultralytics import YOLO

# 변환할 모델 파일명과 이미지 사이즈
models = [
    ("./models/yolo26n.pt", 640),
    ("./models/yolo26n-cls.pt", 224),
    ("./models/yolo26n-pose.pt", 640),
    ("./models/yolo26n-seg.pt", 640),
    ("./models/yolo26n-obb.pt", 640)
]

for model_file, imgsz in models:
    print(f"{model_file} 변환")
    YOLO(model_file).export(format="engine", device="0", imgsz=imgsz, workspace=2)

print("모든 변환 완료")
