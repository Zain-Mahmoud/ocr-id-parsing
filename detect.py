from ultralytics import YOLO

model = YOLO('yolo26n.pt') 

results = model.train('/path/to/training/data') 

results.export('onnx') 