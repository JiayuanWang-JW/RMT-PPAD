from ultralytics import RTDETR

# Load a model
model = RTDETR("/home/jiayuan/code/rtdetr/runs/detect/train4/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="engine")