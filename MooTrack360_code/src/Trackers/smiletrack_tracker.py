import yaml
import numpy as np
# Monkey-patch np.float if it doesn't exist
if not hasattr(np, 'float'):
    np.float = float
from typing import List, Optional
from dataclasses import dataclass
import torch

from Trackers.tracker_models.SMILEtrack.mc_SMILEtrack import SMILEtrack
from .base_tracker import BaseTracker, Track

@dataclass(frozen=True)
class SMILETrackerArgs:
    track_low_thresh: float = 0.1
    track_high_thresh: float = 0.25
    new_track_thresh: float = 0.4
    track_buffer: int = 60
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    with_reid: bool = False
    cmc_method: str = 'sparseOptFlow'
    name: str = 'exp'
    ablation: bool = False

class SMILETrackTracker(BaseTracker):
    """
    A concrete subclass that uses the ByteTrack algorithm.
    """

    def __init__(self, tracker_params_path: str = "VelKoTek/src/configs/tracker_configs.yaml"):
        """
        :param track_thresh: Tracking confidence threshold
        :param high_thresh: High confidence threshold for bounding boxes
        :param match_thresh: Distance threshold for matching
        :param frame_rate: FPS of the video, used internally by ByteTrack
        """
        super().__init__()
        
        with open(tracker_params_path, "r") as f:
                config = yaml.safe_load(f)
        self.smiletrack_args = config["smiletracker"]

        # Create a SMILETrackerArgs instance with the given parameters
        self.args = SMILETrackerArgs(
            track_low_thresh = self.smiletrack_args["track_low_thresh"],
            track_high_thresh = self.smiletrack_args["track_high_thresh"],
            new_track_thresh = self.smiletrack_args["new_track_thresh"],
            track_buffer = self.smiletrack_args["track_buffer"],
            match_thresh = self.smiletrack_args["match_thresh"],
            aspect_ratio_thresh = self.smiletrack_args["aspect_ratio_thresh"],
            min_box_area = self.smiletrack_args["min_box_area"],
            mot20 = self.smiletrack_args["mot20"],
            proximity_thresh = self.smiletrack_args["proximity_thresh"],
            appearance_thresh = self.smiletrack_args["appearance_thresh"],
            with_reid = self.smiletrack_args["with_reid"],
            cmc_method = self.smiletrack_args["cmc_method"],
            name = self.smiletrack_args["name"],
            ablation = self.smiletrack_args["ablation"],
            )

        # Initialize the underlying SMILEtrack instance
        self.tracker = SMILEtrack(self.args)
        self.frame_id = 0

    def update(self, detections: List, frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        Updates SMILEtrack with the current detections and returns the active tracks.

        :param detections: A list of Detection objects with attributes:
                           x1, y1, x2, y2, confidence, and class_id.
        :param frame: (Optional) The current video frame if needed for re-ID or appearance info.
        :return: List[Track] objects
        """
        self.frame_id += 1

        # Get frame dimensions if available.
        if frame is not None:
            H, W, _ = frame.shape
        else:
            H, W = 2944, 2944  # fallback values

        # Convert your detections to SMILEtrack's format: [x, y, w, h, score]
        bboxes = []
        for det in detections:
            # Extract the attributes from the Detection object.
            x1 = det.x1
            y1 = det.y1
            x2 = det.x2
            y2 = det.y2
            score = det.confidence
            cls  = det.class_id  
            
            w = x2 - x1
            h = y2 - y1
            bboxes.append([x1, y1, x2, y2, score, cls])
        bboxes_np = np.asarray(bboxes, dtype=float)

        bboxes_tensor = torch.from_numpy(bboxes_np)

        # SMILEtrack typically also needs some meta info about the frame
        # e.g., "img_size" and "img_info"
        # For simplicity, assume we just pass the required arguments:
        # img_info = (frame.shape[0], frame.shape[1]) if frame is not None else (1080, 1920)
        # update returns a list of track objects

        # Use frame dimensions as image info.
        img_info = (H, W)

        online_targets = self.tracker.update(
            output_results=bboxes_tensor,
            img=frame
        )

        # Convert SMILEtrack's online_targets to our standardized Track objects
        out_tracks = []
        for t in online_targets:
            # Typically t.tlwh gives (x, y, w, h)
            # t.track_id is the ID, t.score is the confidence
            x, y, w, h = t.tlwh
            x2 = x + w
            y2 = y + h
            out_tracks.append(
                Track(
                    track_id=int(t.track_id),
                    x1=float(x),
                    y1=float(y),
                    x2=float(x2),
                    y2=float(y2),
                    score=float(t.score),
                    class_id=int(t.cls)
                )
            )

        return out_tracks

    def reset(self):
        """
        Resets the ByteTrack state and internal counters.
        """
        self.frame_id = 0
        # If there's a method in BYTETracker to fully reset, call it here:
        # self.tracker.reset()
        pass
