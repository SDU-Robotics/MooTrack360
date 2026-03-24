import yaml
import numpy as np
# Monkey-patch np.float if it doesn't exist
if not hasattr(np, 'float'):
    np.float = float
from typing import List, Optional
from dataclasses import dataclass
import torch

from Trackers.tracker_models.ByteTrack.byte_tracker import BYTETracker
from .base_tracker import BaseTracker, Track

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 60
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

class ByteTrackTracker(BaseTracker):
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
        self.bytetrack_args = config["bytetracker"]

        self.frame_rate = self.bytetrack_args["frame_rate"]
        # Create a BYTETrackerArgs instance with the given parameters
        self.args = BYTETrackerArgs(
            track_thresh=self.bytetrack_args["track_thresh"],
            track_buffer=self.bytetrack_args["track_buffer"],
            match_thresh=self.bytetrack_args["match_thresh"],
            aspect_ratio_thresh=self.bytetrack_args["aspect_ratio_thresh"],
            min_box_area=self.bytetrack_args["min_box_area"],
            mot20= self.bytetrack_args["mot20"])

        # Initialize the underlying ByteTrack instance
        self.tracker = BYTETracker(self.args,self.frame_rate)
        self.frame_id = 0

    def update(self, detections: List, frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        Updates ByteTrack with the current detections and returns the active tracks.

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

        # Convert your detections to ByteTrack's format: [x, y, w, h, score]
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

        # ByteTrack typically also needs some meta info about the frame
        # e.g., "img_size" and "img_info"
        # For simplicity, assume we just pass the required arguments:
        # img_info = (frame.shape[0], frame.shape[1]) if frame is not None else (1080, 1920)
        # update returns a list of track objects

        # Use frame dimensions as image info.
        img_info = (H, W)

        online_targets = self.tracker.update(
            output_results=bboxes_tensor,
            img_info=img_info,
            img_size=img_info,
            #frame_id=self.frame_id
        )

        # Convert ByteTrack's online_targets to our standardized Track objects
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
                    class_id=int(t.class_id)
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
