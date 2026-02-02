# motiondetector/analytics/yolo_pose.py
import logging
import torch

logger = logging.getLogger(__name__)


class YoloPose:
    def __init__(self, cuda_id: int, model: str, conf: float = 0.5):
        self.cuda_id = int(cuda_id)
        self.model = model
        self.conf = float(conf)
        self.detector = None

    def load(self):
        from ultralytics import YOLO
        assert torch.cuda.is_available(), "CUDA is not available"
        assert self.cuda_id < torch.cuda.device_count(), f"GPU out of index: {self.cuda_id}"
        torch.cuda.set_device(self.cuda_id)

        self.detector = YOLO(self.model)
        self.detector.to(f'cuda:{self.cuda_id}')
        logger.info("YOLO Pose loaded on cuda:%s model=%s", self.cuda_id, self.model)

    def track(self, frame, **kwargs):
        return self.detector.track(frame, **kwargs)

    def predict(self, frame, **kwargs):
        return self.detector.predict(frame, **kwargs)
