from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
# model = RTDETR("/home/jiayuan/code/rtdetr/runs/detect/train4/weights/best.pt")
model = RTDETR("/home/jiayuan/code/rtdetr/runs/detect/train4/weights/best.pt")

# Display model information (optional)
# model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.val(data="/home/jiayuan/code/rtdetr/ultralytics/cfg/datasets/BDD.yaml", imgsz=640, batch=8, device=[1])

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")