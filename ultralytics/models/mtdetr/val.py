# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data import YOLODataset, converter
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops
from ultralytics.utils.metrics import DetMetrics, SegmentMetrics, ConfusionMatrix, SegmentationMetric, AverageMeter
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import output_to_target, plot_images
import os
from pathlib import Path
import numpy as np
import torch.nn.functional as F

__all__ = ("MTDETRValidator",)  # tuple or list


class MTDETRDataset(YOLODataset):
    """
    Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.
    """

    def __init__(self, *args, data=None, **kwargs):
        """Initialize the RTDETRDataset class by inheriting from the YOLODataset class."""
        super().__init__(*args, data=data, **kwargs)

    # NOTE: add stretch version load_image for RTDETR mosaic
    def load_image(self, i, rect_mode=False):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """Temporary, only for evaluation."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleFill=True)])
            transforms = Compose([])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                merge_mask=True,
                imgsz=self.imgsz,
                seg_nc=len(self.data['type_task']['segmentation']),
            )
        )
        return transforms


class MTDETRValidator(DetectionValidator):
    """
    RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies Non-maximum suppression for
    post-processing, and updates evaluation metrics accordingly.

    Example:
        ```python
        from ultralytics.models.rtdetr import RTDETRValidator

        args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        validator = RTDETRValidator(args=args)
        validator()
        ```

    Note:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """

    def __init__(self, dataloader=None, number_task=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.number_task = number_task
        self.plot_masks = None
        self.process = None
        self.args.task = "multi"
        ### JW init two metics for different task.
        self.metrics = {'detection': DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot), 'segmentation': SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)}
        if not isinstance(self.args.mask_threshold, list):
            self.args.mask_threshold = [self.args.mask_threshold]
        self.mask_thr = torch.tensor(self.args.mask_threshold).view(1, len(self.args.mask_threshold), 1, 1)   ### JW set the mask_threshold

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(len(model.names)))
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = self.number_task
        self.filtered_names = {task: {i: self.names[i] for i in indices if i in self.names} for task, indices in self.data['type_task'].items()}
        self.metrics['detection'].names = self.filtered_names['detection']
        self.metrics['detection'].plot = self.args.plots
        self.metrics['segmentation'].names = self.filtered_names['segmentation']
        self.metrics['segmentation'].plot = self.args.plots
        ### JW init the segmentation task
        self.seg_metrics = {self.dataloader.dataset.data['names'][i]: SegmentationMetric(2) for i in self.dataloader.dataset.data['type_task']['segmentation']}
        self.seg_result = {self.dataloader.dataset.data['names'][i]: {'pixacc': AverageMeter(), 'subacc': AverageMeter(), 'IoU': AverageMeter(),
                                  'mIoU': AverageMeter()} for i in self.dataloader.dataset.data['type_task']['segmentation']}
        self.plot_masks = {self.dataloader.dataset.data['names'][i]: [] for i in self.dataloader.dataset.data['type_task']['segmentation']}
        self.confusion_matrix_detection = ConfusionMatrix(nc=self.nc['detection'], conf=self.args.conf)
        self.confusion_matrix_segmentation = ConfusionMatrix(nc=self.nc['segmentation'], conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        # batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return MTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            # Do not need threshold for evaluation as only got 300 boxes here
            # idx = score > self.args.conf
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            # Sort by confidence to correctly get internal metrics
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred  # [idx]

        mask_thr_tensor = self.mask_thr.to(device=preds[1][5].device)
        mask = (torch.sigmoid(preds[1][5]) > mask_thr_tensor).float()

        return outputs, mask

    def _prepare_batch(self, si, batch, detection_indices):
        """Prepares a batch for training or inference by applying transformations."""
        idx_detection = detection_indices[batch["batch_idx"][detection_indices] == si]
        cls_detection = batch["cls"][idx_detection].squeeze(-1)
        bbox_detection = batch["bboxes"][idx_detection]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls_detection):
            bbox_detection = ops.xywh2xyxy(bbox_detection)  # target boxes
            bbox_detection[..., [0, 2]] *= ori_shape[1]  # native-space pred
            bbox_detection[..., [1, 3]] *= ori_shape[0]  # native-space pred
        return {"cls": cls_detection, "bbox": bbox_detection, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch with transformed bounding boxes and class labels."""
        predn = pred.clone()
        predn[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # native-space pred
        predn[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # native-space pred
        return predn.float()

    def update_metrics(self, preds_list, batch):
        """Metrics."""
        preds = preds_list[0]
        merge_mask = preds_list[1]
        type_task = batch['type_task'][0]
        detection_classes = torch.tensor(type_task['detection'])
        detection_indices = torch.where(torch.isin(batch["cls"], detection_classes.to(device=batch["cls"].device)))[0]
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )

            pbatch_detection= self._prepare_batch(si, batch, detection_indices)
            cls, bbox = pbatch_detection.pop("cls"), pbatch_detection.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch_detection)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            ### JW segmentataion evaluate
            _, nc, _, _ = merge_mask.shape
            for seg_nc in range(nc):
                task_name = self.data['names'][self.data['type_task']['segmentation'][seg_nc]]
                pred_mask = merge_mask[si][seg_nc].squeeze()
                gt_mask = batch['merge_mask'][si][seg_nc].to(device=pred_mask.device)
                self.seg_metrics[task_name].reset()

                if gt_mask.shape != pred_mask.shape:
                    pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
                    gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 320, 320)
                    pred_mask = F.interpolate(
                        pred_mask,
                        size=gt_mask.shape[-2:],
                        mode='bilinear',
                    )
                    pred_mask = pred_mask.squeeze(0).squeeze(0)
                    gt_mask = gt_mask.squeeze(0).squeeze(0)

                self.seg_metrics[task_name].addBatch(pred_mask.cpu(), gt_mask.cpu())
                self.seg_result[task_name]['pixacc'].update(self.seg_metrics[task_name].pixelAccuracy())
                self.seg_result[task_name]['subacc'].update(self.seg_metrics[task_name].lineAccuracy())
                self.seg_result[task_name]['IoU'].update(self.seg_metrics[task_name].IntersectionOverUnion())
                self.seg_result[task_name]['mIoU'].update(self.seg_metrics[task_name].meanIntersectionOverUnion())
                if self.args.plots:
                    self.plot_masks[task_name].append(pred_mask.cpu())  ### JW TODO Need to adapt to the segmentation task plot
            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix_detection.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch_detection["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc['detection'])
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc['detection'])
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics['detection'].process(**stats)
        return self.metrics['detection'].results_dict

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics['detection'].confusion_matrix = self.confusion_matrix

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics['detection'].keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics['detection'].mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc['detection'] > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics['detection'].class_result(i))
                )

        pf = '%22s' + ('%11s' + '%11.3g') * 4
        seg_index = self.data['type_task']['segmentation']
        for i in seg_index:
            key_values = [(key, value.avg) for key, value in self.seg_result[self.data['names'][i]].items()]
            LOGGER.info(pf % (self.data['names'][i], *sum(key_values, ())))

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix_detection.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred