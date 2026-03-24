import yaml
import numpy as np
import torch
from typing import List, Optional, Dict
from dataclasses import dataclass
import types

# Base interface and Track data structure
from .base_tracker import BaseTracker, Track

# ConfTrack imports
from Trackers.tracker_models.ConfTrack.base_tracker import BaseTracker as ConfTrackClass
from Trackers.tracker_models.ConfTrack.detection.base_detection import BaseDetection

@dataclass(frozen=True)
class ConfTrackTrackerArgs:
    use_extractor: bool = False
    use_cmc: bool = False
    # You can add more ConfTrack-specific parameters here

class ConfTrackTracker(BaseTracker):
    """
    Wrapper for the ConfTrack multi-object tracker.
    Carries through detection class and score via the matching output.
    """
    def __init__(
        self,
        tracker_params_path: str = "VelKoTek/src/configs/tracker_configs.yaml",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        # Load YAML config
        with open(tracker_params_path, "r") as f:
            cfg = yaml.safe_load(f)
        ct_cfg_dict = cfg.get("conftrack", {})

        # Convert dict to an object with attributes
        trk_cfg = types.SimpleNamespace(**ct_cfg_dict)
        trk_cfg.device = device  # add device to config

        # type_matching is needed by get_matching_fn
        trk_cfg.type_matching = ct_cfg_dict.get("type_matching", "conftrack")
        trk_cfg.track_new_thr = ct_cfg_dict.get("track_new_thr", 0.5)
        trk_cfg.use_extractor = ct_cfg_dict.get("use_extractor", False)
        trk_cfg.use_cmc = ct_cfg_dict.get("use_cmc", False)
        trk_cfg.detection_high_thr = ct_cfg_dict.get("detection_high_thr", trk_cfg.track_new_thr)  # for matching stages

        # Store args
        self.args = ConfTrackTrackerArgs(
            use_extractor = trk_cfg.use_extractor,
            use_cmc       = trk_cfg.use_cmc
        )

        # Instantiate ConfTrack tracker with proper config object
        self.tracker = ConfTrackClass(trk_cfg, device=device)
        self.tracker.initialize()

        self.frame_id = 0
        # store last seen class and score per track_id
        self.track_classes: Dict[int, int] = {}
        self.track_scores:  Dict[int, float] = {}

    def update(self, detections: List, frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        :param detections: list of detection objects with x1,y1,x2,y2,confidence,class_id
        :param frame:      optional raw frame passed to extractor/CMC
        :returns:          list of Track(track_id, x1,y1,x2,y2, score, class_id)
        """
        self.frame_id += 1

        # Convert to BaseDetection list
        base_dets: List[BaseDetection] = []
        det_bboxes = []
        det_classes = []
        det_scores = []
        for d in detections:
            bbox_arr = np.array([d.x1, d.y1, d.x2, d.y2], dtype=np.float32)
            base_dets.append(BaseDetection(bbox_arr, d.confidence, d.class_id))
            det_bboxes.append([d.x1, d.y1, d.x2, d.y2])
            det_classes.append(d.class_id)
            det_scores.append(d.confidence)
        det_bboxes = np.array(det_bboxes, dtype=np.float32)

        # Predict step (CMC may use frame)
        self.tracker.predict(raw_frame=frame, detections=base_dets, img_idx=self.frame_id)

        # Update step (extractor uses frame if enabled)
        img_for_extractor = None
        if self.args.use_extractor and frame is not None:
            img_for_extractor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
        matches, unmatched_trks, unmatched_dets, deleted_trks = \
            self.tracker.update(base_dets, img_for_extractor=img_for_extractor)

        # IoU helper
        def iou(a, b):
            xa1, ya1, xa2, ya2 = a
            xb1, yb1, xb2, yb2 = b
            xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
            xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter + 1e-6
            return inter / union

        # Build output tracks
        out_tracks: List[Track] = []
        # Map matched track index to detection index
        det_map = {trk_idx: det_idx for trk_idx, det_idx in matches}

        for idx, trk in enumerate(self.tracker.tracks):
            tid = trk.track_id
            # Extract bbox via ConfTrack API
            if hasattr(trk, 'get_xyxy'):
                xyxy = trk.get_xyxy()
                x1, y1, x2, y2 = xyxy.tolist()
            elif hasattr(trk, 'to_tlbr'):
                x1, y1, x2, y2 = trk.to_tlbr()
            elif hasattr(trk, 'bbox'):
                x1, y1, x2, y2 = trk.bbox
            else:
                # fallback: use projected Kalman state
                proj = trk.get_projected_state()
                cx, cy, w, h = proj[0], proj[1], proj[2], proj[3]
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2

            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

            # Match back to original detection if available
            if idx in det_map and det_bboxes.size > 0:
                det_idx = det_map[idx]
                cls = det_classes[det_idx]
                sc  = det_scores[det_idx]
            else:
                cls = self.track_classes.get(tid, 0)
                sc  = self.track_scores.get(tid, 1.0)

            # store for next frame
            self.track_classes[tid] = cls
            self.track_scores[tid] = sc

            out_tracks.append(
                Track(
                    track_id=tid,
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    score=float(sc),
                    class_id=int(cls)
                )
            )
        return out_tracks


    def reset(self):
        self.frame_id = 0
        self.tracker.initialize()  # resets tracks and IDs