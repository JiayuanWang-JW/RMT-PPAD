# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Interface for Baidu's RT-DETR, a Vision Transformer-based real-time object detector. RT-DETR offers real-time
performance and high accuracy, excelling in accelerated backends like CUDA with TensorRT. It features an efficient
hybrid encoder and IoU-aware query selection for enhanced detection accuracy.

For more information on RT-DETR, visit: https://arxiv.org/pdf/2304.08069.pdf
"""

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import MTDETRModel

from .predict import MTDETRPredictor
from .train import MTDETRTrainer
from .val import MTDETRValidator


class MTDETR(Model):
    """
    Interface for Baidu's RT-DETR model. This Vision Transformer-based object detector provides real-time performance
    with high accuracy. It supports efficient hybrid encoding, IoU-aware query selection, and adaptable inference speed.

    Attributes:
        model (str): Path to the pre-trained model. Defaults to 'rtdetr-l.pt'.
    """

    def __init__(self, model="mtdetr-l.pt") -> None:
        """
        Initializes the RT-DETR model with the given pre-trained model file. Supports .pt and .yaml formats.

        Args:
            model (str): Path to the pre-trained model. Defaults to 'rtdetr-l.pt'.

        Raises:
            NotImplementedError: If the model file extension is not 'pt', 'yaml', or 'yml'.
        """
        super().__init__(model=model, task="multi")

    @property
    def task_map(self) -> dict:
        """
        Returns a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            dict: A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
        return {
            "multi": {
                "predictor": MTDETRPredictor,
                "validator": MTDETRValidator,
                "trainer": MTDETRTrainer,
                "model": MTDETRModel,
            }
        }
