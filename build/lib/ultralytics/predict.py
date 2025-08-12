from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
# model = RTDETR("/home/jiayuan/code/rtdetr/runs/detect/train4/weights/best.pt")
model = RTDETR("/home/jiayuan/code/rtdetr/runs/detect/train4/weights/best.engine")

# Display model information (optional)
# model.info()

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.val(data="/home/jiayuan/code/rtdetr/ultralytics/cfg/datasets/BDD.yaml", imgsz=640, batch=1, device=[0])

model.predict(source='/home/jiayuan/data/BDD_vehicle/images/val2017', imgsz=(640,640), device=[0],name='predict', conf=0.25, iou=0.45, show_labels=False, save=True)

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")