from abc import ABC, abstractmethod
from typing import List

class Detection:
    """
    A simple container for detection results.
    You could also make this a @dataclass or namedtuple.
    """
    def __init__(self, x1: float, y1: float, x2: float, y2: float,
                 confidence: float, class_id: int):
        """
        :param x1, y1: top-left corner of bounding box
        :param x2, y2: bottom-right corner of bounding box
        :param confidence: detection confidence (0 to 1)
        :param class_id: numeric class id (e.g., 0 for 'cow')
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_id = class_id

    def __repr__(self):
        return (f"Detection(x1={self.x1}, y1={self.y1}, "
                f"x2={self.x2}, y2={self.y2}, "
                f"conf={self.confidence}, class_id={self.class_id})")


class BaseDetector(ABC):
    """
    Abstract base class defining a detector interface.
    """

    @abstractmethod
    def detect_objects(self, frame) -> List[Detection]:
        """
        Given a frame (e.g., a NumPy array representing an image),
        return a list of Detection objects.
        """
        raise NotImplementedError
