from ultralytics import MTDETR
import gc

import time


model = MTDETR("/home/jiayuan/code/RMT-PPAD/ultralytics/cfg/models/mt-detr/rtdetr-l_bdd.yaml")


model.info()

results = model.train(data="/home/jiayuan/code/RMT-PPAD/ultralytics/cfg/datasets/BDD_full.yaml", epochs=250, batch=45, device=[0,1,2], task='multi', overlap_mask=False, mask_ratio=1, plots=False, mask_threshold=[0.45,0.9], deterministic=False, val_period=10, save_period=10, cos_lr=True)