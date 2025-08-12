from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("/home/jiayuan/code/rtdetr/ultralytics/cfg/models/rt-detr/rtdetr-resnet50-BDD.yaml")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/home/jiayuan/code/rtdetr/ultralytics/cfg/datasets/BDD.yaml", epochs=100, imgsz=640, batch=78, device=[0,1,2])

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")