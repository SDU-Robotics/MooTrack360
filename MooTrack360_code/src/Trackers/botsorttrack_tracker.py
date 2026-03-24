import yaml
import numpy as np
# Monkey-patch np.float if it doesn't exist
if not hasattr(np, 'float'):
    np.float = float

import collections
import collections.abc

# If 'Mapping' isn't in collections, add it from collections.abc
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping

import torch
import types

import sys, types

# Ensure torch._six is available for modules that need it.
if "torch._six" not in sys.modules:
    mod = types.ModuleType("torch._six")
    mod.string_classes = (str,)
    sys.modules["torch._six"] = mod

if not hasattr(torch, "_six"):
    torch._six = types.ModuleType("torch._six")
    torch._six.string_classes = (str,)

from typing import List, Optional
from dataclasses import dataclass

# IMPORTANT: Adjust the import to match your Bot-SORT submodule structure.
from Trackers.tracker_models.BoTSORT.bot_sort import BoTSORT
from .base_tracker import BaseTracker, Track

@dataclass(frozen=True)
class BotSORTArgs:
    track_high_thresh: float = 0.6      # High score threshold for association
    track_low_thresh: float = 0.2       # Low score threshold for detection filtering
    new_track_thresh: float = 0.5       # Threshold for initializing a new track
    track_buffer: float = 30.0          # Buffer size (in frames) for lost tracks
    proximity_thresh: float = 0.8       # IoU (or other metric) threshold for matching
    appearance_thresh: float = 0.5      # Appearance distance threshold for re-identification
    with_reid: bool = False             # Whether to use a ReID module
    fast_reid_config: str = ""          # Path to FastReID config file
    fast_reid_weights: str = ""         # Path to FastReID weights
    device: str = "cpu"                 # Device for ReID inference (e.g. "cpu" or "cuda")
    cmc_method: str = "default"         # Method name for camera motion compensation (CMC)
    name: str = "botsort"               # Tracker name (used in verbosity/debug)
    ablation: bool = False              # Ablation flag for experiments
    mot20: bool = False                 # Whether to use MOT20 evaluation criteria
    match_thresh: float = 0.7           # Threshold for assignment in linear_assignment

class BotSortTracker(BaseTracker):
    """
    A concrete subclass that uses the BOT-SORT algorithm.
    """

    def __init__(self, tracker_params_path: str = "VelKoTek/src/configs/tracker_configs.yaml"):
        """
        Initializes BOT-SORT with parameters read from a YAML file.
        """
        super().__init__()
        with open(tracker_params_path, "r") as f:
            config = yaml.safe_load(f)
        self.botsort_args = config["botsort"]

        self.frame_rate = self.botsort_args["frame_rate"]
        # Create a BotSORTArgs instance using parameters from the config.
        self.args = BotSORTArgs(
            track_high_thresh=self.botsort_args["track_high_thresh"],
            track_low_thresh=self.botsort_args["track_low_thresh"],
            new_track_thresh=self.botsort_args["new_track_thresh"],
            track_buffer=self.botsort_args["track_buffer"],
            proximity_thresh=self.botsort_args["proximity_thresh"],
            appearance_thresh=self.botsort_args["appearance_thresh"],
            with_reid=self.botsort_args["with_reid"],
            fast_reid_config=self.botsort_args["fast_reid_config"],
            fast_reid_weights=self.botsort_args["fast_reid_weights"],
            device=self.botsort_args["device"],
            cmc_method=self.botsort_args["cmc_method"],
            name=self.botsort_args["name"],
            ablation=self.botsort_args["ablation"],
            mot20=self.botsort_args["mot20"],
            match_thresh=self.botsort_args["match_thresh"]
        )

        # Initialize the underlying BOT-SORT instance.
        # (Ensure that the BOTSort class is implemented with a similar interface to BYTETracker.)
        self.tracker = BoTSORT(self.args, self.frame_rate)
        self.frame_id = 0

    def update(self, detections: List, frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        Updates BOT-SORT with the current detections and returns the active tracks.

        :param detections: A list of Detection objects with attributes:
                           x1, y1, x2, y2, confidence, and class_id.
        :param frame: (Optional) The current video frame if needed for appearance info.
        :return: List[Track] objects.
        """
        self.frame_id += 1

        # Get frame dimensions if available.
        if frame is not None:
            H, W, _ = frame.shape
        else:
            H, W = 2944, 2944  # fallback values

        # Convert your detections to BOT-SORT's expected format in TLBR form:
        # [x1, y1, x2, y2, score, class_id]
        bboxes = []
        for det in detections:
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
            score = det.confidence
            cls  = det.class_id
            # (Assumes detections are already in pixel coordinates.)
            bboxes.append([x1, y1, x2, y2, score, cls])
        bboxes_np = np.asarray(bboxes, dtype=float)

        # Convert the NumPy array to a torch tensor.
        bboxes_tensor = torch.from_numpy(bboxes_np)

        # Use frame dimensions as image info.
        img_info = (H, W)

        # Call BOT-SORT's update method.
        online_targets = self.tracker.update(
            output_results=bboxes_tensor,
            img=frame
        )

        # Convert BOT-SORT's output to our standardized Track objects.
        out_tracks = []
        for t in online_targets:
            # Assume t.tlwh is computed internally from TLBR: [x, y, w, h]
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
        Resets the BOT-SORT state and internal counters.
        """
        self.frame_id = 0
        if hasattr(self.tracker, 'reset'):
            self.tracker.reset()
