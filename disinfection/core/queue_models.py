# motiondetector/core/queue_models.py
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass
class FrameItem:
    ts_ms: int
    frame: np.ndarray


@dataclass
class InferenceItem:
    ts_ms: int
    frame: np.ndarray
    results: Any


@dataclass
class IOTask:
    upload: bool = False
    image: Optional[np.ndarray] = None
    person_id: Any = None
    timestamp: Any = None
    save_count: int = 0
