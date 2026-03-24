import yaml
import numpy as np
import torch
from typing import List, Optional
from dataclasses import dataclass

# Assuming the official SORT code is in Trackers/tracker_models/SORT.sort
from Trackers.tracker_models.SORT.sort import Sort

# Base classes
from .base_tracker import BaseTracker, Track

@dataclass(frozen=True)
class SortArgs:
    max_age: int = 1
    min_hits: int = 3
    iou_threshold: float = 0.3

class SortTracker(BaseTracker):
    """
    A concrete subclass that uses the SORT algorithm (no ReID).
    """

    def __init__(self, tracker_params_path: str = "VelKoTek/src/configs/tracker_configs.yaml"):
        super().__init__()
        
        # Load config
        with open(tracker_params_path, "r") as f:
            config = yaml.safe_load(f)
        self.sort_args = config["sort"]  # Adjust key if needed

        # Create a SortArgs dataclass
        self.args = SortArgs(
            max_age=self.sort_args.get("max_age", 1),
            min_hits=self.sort_args.get("min_hits", 3),
            iou_threshold=self.sort_args.get("iou_threshold", 0.3),
        )

        # Initialize SORT
        self.tracker = Sort(
            max_age=self.args.max_age,
            min_hits=self.args.min_hits,
            iou_threshold=self.args.iou_threshold
        )

        self.frame_id = 0

    def update(self, detections: List, frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        Updates SORT with the current detections and returns active tracks.

        :param detections: A list of Detection objects with attributes:
                           x1, y1, x2, y2, confidence, (and optional class_id)
        :param frame: (Optional) The current video frame; SORT does not need it.
        :return: A list of `Track` objects.
        """
        self.frame_id += 1

        # Convert to [x1, y1, x2, y2, score]
        bboxes = []
        for det in detections:
            bboxes.append([det.x1, det.y1, det.x2, det.y2, det.confidence, det.class_id])

        # Convert to NumPy or torch (SORT update uses NumPy arrays)
        bboxes_np = np.array(bboxes, dtype=np.float32)

        # Run SORT's update -> Nx5, with columns [x1, y1, x2, y2, track_id]
        tracked_objects = self.tracker.update(bboxes_np)

        # Convert tracked_objects into our standardized Track format
        out_tracks = []
        for row in tracked_objects:
            x1, y1, x2, y2, score, class_id, track_id = row
            # SORT does not maintain a track confidence; set your own placeholder if needed
            out_tracks.append(
                Track(
                    track_id=int(track_id),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=score,
                    class_id=class_id  # Or parse from detection if you want to keep classes
                )
            )
        return out_tracks

    def reset(self):
        """
        Resets SORT’s state and internal counters.
        """
        self.frame_id = 0
        self.tracker = Sort(
            max_age=self.args.max_age,
            min_hits=self.args.min_hits,
            iou_threshold=self.args.iou_threshold
        )
