# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import MTDETRModel
from ultralytics.utils import RANK, colorstr

from .val import MTDETRDataset, MTDETRValidator


class MTDETRTrainer(DetectionTrainer):
    """
    Trainer class for the RT-DETR model developed by Baidu for real-time object detection. Extends the DetectionTrainer
    class for YOLO to adapt to the specific features and architecture of RT-DETR. This model leverages Vision
    Transformers and has capabilities like IoU-aware query selection and adaptable inference speed.

    Notes:
        - F.grid_sample used in RT-DETR does not support the `deterministic=True` argument.
        - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.

    Example:
        ```python
        from ultralytics.models.rtdetr.train import RTDETRTrainer

        args = dict(model="rtdetr-l.yaml", data="coco8.yaml", imgsz=640, epochs=3)
        trainer = RTDETRTrainer(overrides=args)
        trainer.train()
        ```
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (dict, optional): Model configuration. Defaults to None.
            weights (str, optional): Path to pre-trained model weights. Defaults to None.
            verbose (bool): Verbose logging if True. Defaults to True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """
        model = MTDETRModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build and return an RT-DETR dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size for rectangle training. Defaults to None.

        Returns:
            (RTDETRDataset): Dataset object for the specific mode.
        """
        return MTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def get_validator(self):
        """
        Returns a DetectionValidator suitable for RT-DETR model validation.

        Returns:
            (RTDETRValidator): Validator object for model validation.
        """
        self.loss_names_det = "giou_loss", "cls_loss", "l1_loss" ### JW TBD, require to add the segmentation loss
        self.loss_names_seg = "TBD", "TBD"
        self.loss_names = "Detection", "da_Seg", "ll_seg"
        self.loss_diy = "Det_gate", "Seg_gate"

        ### JW collect the number of each task.
        number_task = {key: len(value) for key, value in self.data['type_task'].items()}

        return MTDETRValidator(self.test_loader, number_task, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix="train", task=None):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        if task is 'detection':
            keys = [f"{prefix}/{x}" for x in self.loss_names_det]
        else:
            keys = [f"{prefix}/{x}" for x in self.loss_names_seg]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images. Scales and converts the images to float format.

        Args:
            batch (dict): Dictionary containing a batch of images, bboxes, and labels.

        Returns:
            (dict): Preprocessed batch.
        """
        batch = super().preprocess_batch(batch)
        bs = len(batch["img"])
        batch_idx = batch["batch_idx"]
        gt_bbox, gt_class, gt_mask = [], [], []
        for i in range(bs):
            gt_bbox.append(batch["bboxes"][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch["cls"][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
            # gt_mask.append(batch["masks"][batch_idx == i].to(batch_idx.device))  ### JW add ground truth for mask to batch
        return batch

    def progress_string(self):
        return ("\n" + "%11s" * (6 + len(self.loss_names) +len(self.loss_diy))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "lr",
            "grad_norm",
            *self.loss_diy,
            "Instances",
            "Size",
        )
