from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

class Track:
    """
    A container for the results of the tracking step.
    """
    def __init__(self, track_id: int, x1: float, y1: float, x2: float, y2: float, score: float = 1.0, class_id: int = 0):
        """
        :param track_id: Unique ID assigned by the tracker
        :param x1, y1: Top-left corner of bounding box
        :param x2, y2: Bottom-right corner of bounding box
        :param score: Confidence score for the track (varies by tracker implementation)
        """
        self.track_id = track_id
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score  # or any other metric the tracker might output
        self.class_id = class_id

    def bbox_tlwh(self):
        """Return bbox in (x1, y1, width, height) format."""
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def bbox_tlbr(self):
        """Return bbox in (x1, y1, x2, y2) format."""
        return (self.x1, self.y1, self.x2, self.y2)

    def __repr__(self):
        return (f"Track(id={self.track_id}, bbox_tlbr={self.bbox_tlbr()}, "
                f"score={self.score:.2f}, class_id={self.class_id})")
    


class BaseTracker(ABC):
    """
    Abstract base class defining a high-level interface for multi-object trackers.
    """

    @abstractmethod
    def update(self, detections: List, frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        Updates the tracker state with the current frame's detections and returns the active tracks.
        
        :param detections: A list of detection objects or bounding boxes from your detector.
                           The structure of each detection depends on how you parse them,
                           but typically includes (x1, y1, x2, y2, confidence, class_id, ...).
        :param frame: (Optional) The current video frame (if needed by the tracker for appearance features).
        :return: A list of Track objects representing tracked objects after update.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Resets the state of the tracker. This can be used when starting a new sequence
        or if you want to clear all tracks.
        """
        raise NotImplementedError
