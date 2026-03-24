import yaml
import numpy as np
import torch
from typing import List, Optional
from dataclasses import dataclass

# Local imports
from .base_tracker import BaseTracker, Track

# OC-SORT import (assuming you have your OCSort code under Trackers/tracker_models/OC_SORT)
from Trackers.tracker_models.OcSORT.ocsort import OCSort

@dataclass(frozen=True)
class OCSortArgs:
    det_thresh: float = 0.5
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    delta_t: int = 3
    asso_func: str = "iou"
    inertia: float = 0.2
    use_byte: bool = False

class OCSortTracker(BaseTracker):
    """
    A concrete subclass that uses the OC-SORT algorithm.
    """
    def __init__(self, tracker_params_path: str = "VelKoTek/src/configs/tracker_configs.yaml"):
        super().__init__()
        
        # Load config
        with open(tracker_params_path, "r") as f:
            config = yaml.safe_load(f)
        self.ocsort_args = config["ocsort"]  # Adjust to match your actual YAML structure

        # Create OCSortArgs instance
        self.args = OCSortArgs(
            det_thresh=self.ocsort_args.get("det_thresh", 0.5),
            max_age=self.ocsort_args.get("max_age", 30),
            min_hits=self.ocsort_args.get("min_hits", 3),
            iou_threshold=self.ocsort_args.get("iou_threshold", 0.3),
            delta_t=self.ocsort_args.get("delta_t", 3),
            asso_func=self.ocsort_args.get("asso_func", "iou"),
            inertia=self.ocsort_args.get("inertia", 0.2),
            use_byte=self.ocsort_args.get("use_byte", False),
        )

        # Initialize the underlying OCSort instance
        self.tracker = OCSort(
            det_thresh=self.args.det_thresh,
            max_age=self.args.max_age,
            min_hits=self.args.min_hits,
            iou_threshold=self.args.iou_threshold,
            delta_t=self.args.delta_t,
            asso_func=self.args.asso_func,
            inertia=self.args.inertia,
            use_byte=self.args.use_byte
        )
        self.frame_id = 0

    def update(self, detections: List, frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        Updates OC-SORT with the current detections and returns active tracks.

        :param detections: A list of Detection objects with attributes:
                           x1, y1, x2, y2, confidence, class_id (if needed)
        :param frame: (Optional) The current video frame (not strictly required for OC-SORT)
        :return: List[Track] objects
        """
        self.frame_id += 1

        # If frame is known, get dimensions; else fallback
        if frame is not None:
            H, W, _ = frame.shape
        else:
            # Fallback – adjust to your data if needed
            H, W = 1080, 1920
        
        # Prepare arrays for bounding boxes, scores, and class IDs.
        bboxes = []
        scores = []
        cates = []
        for det in detections:
            bboxes.append([det.x1, det.y1, det.x2, det.y2])
            scores.append(det.confidence)
            cates.append(det.class_id)

        # Convert lists to numpy arrays
        dets_np = np.asarray(bboxes, dtype=np.float32)
        scores_np = np.asarray(scores, dtype=np.float32)
        cates_np = np.asarray(cates)

        # Call update_public() which returns 7 values per tracklet
        online_targets = self.tracker.update_public(dets_np, cates_np, scores_np)

        out_tracks = []
        for row in online_targets:
            # Unpack in the expected order: x1, y1, x2, y2, track_id, class_id, score
            x1, y1, x2, y2, track_id, class_id, score = row
            out_tracks.append(
                Track(
                    track_id=int(track_id),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=score,
                    class_id=class_id
                )
            )
        return out_tracks

    def reset(self):
        """
        Resets the OC-SORT state and internal counters.
        """
        self.frame_id = 0
        # Reinitialize OCSort (there's no built-in reset method)
        self.tracker = OCSort(
            det_thresh=self.args.det_thresh,
            max_age=self.args.max_age,
            min_hits=self.args.min_hits,
            iou_threshold=self.args.iou_threshold,
            delta_t=self.args.delta_t,
            asso_func=self.args.asso_func,
            inertia=self.args.inertia,
            use_byte=self.args.use_byte
        )
