from ultralytics import MTDETR

model = MTDETR("/home/jiayuan/code/RMT-PPAD/ultralytics/best.pt")

results = model.val(data="/home/jiayuan/code/RMT-PPAD/ultralytics/cfg/datasets/BDD_full.yaml", gc=False, imgsz=640, mask_ratio=1, overlap_mask=False, batch=1, device=[0], mask_threshold=[0.45,0.9], plots=False, project="runs", name="RMT-PPAD")

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")