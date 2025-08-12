from ultralytics import MTDETR

# Load a model
model = MTDETR("/home/jiayuan/code/mtdetr/runs/multi/train2002/weights/epoch200.pt")
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(resume=True, data="/home/jiayuan/code/mtdetr/ultralytics/cfg/datasets/BDD_1Wtoy_withmask.yaml", layers=27, gc=True, epochs=201, batch=12, device='cpu', task='multi', overlap_mask=False, mask_ratio=1, plots=False, mask_threshold=[0.45,0.9], deterministic=False, val_period=5, save_period=20,cos_lr=True)  #


