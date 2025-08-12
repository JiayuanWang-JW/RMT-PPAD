from ultralytics import MTDETR


model = MTDETR("/home/jiayuan/code/RMT-PPAD/ultralytics/best.pt")

model.predict(source='/home/jiayuan/data/dash_camara_dataset/night', imgsz=(640,640), device=[1], mask_threshold=[0.45,0.9], show_labels=False, save=True, project="runs", name="predict_night")

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")