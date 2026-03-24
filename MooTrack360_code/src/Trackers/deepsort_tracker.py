import yaml
import numpy as np
import torch
from typing import List, Optional, Dict
from dataclasses import dataclass

# Base interface and Track data structure
from .base_tracker import BaseTracker, Track

# DeepSort imports
from Trackers.tracker_models.DEEPSort.nn_matching import NearestNeighborDistanceMetric
from Trackers.tracker_models.DEEPSort.detection import Detection
from Trackers.tracker_models.DEEPSort.tracker import Tracker as DeepSortTrackerClass

@dataclass(frozen=True)
class DeepSortTrackerArgs:
    metric: str = "cosine"
    matching_threshold: float = 0.2
    budget: Optional[int] = None
    max_iou_distance: float = 0.7
    max_age: int = 30
    n_init: int = 3
    det_conf: float = 0.5

class DeepSortTracker(BaseTracker):
    """
    Wrapper for DeepSort multi-object tracker.
    Input: list of detection objects with x1,y1,x2,y2,confidence (and optionally class_id).
    Optionally uses a re-identification model to extract appearance features.
    Outputs: list of Track(track_id, x1,y1,x2,y2, score, class_id).
    """
    def __init__(
        self,
        tracker_params_path: str = "VelKoTek/src/configs/tracker_configs.yaml",
        reid_model=None
    ):
        super().__init__()
        # Load YAML config
        with open(tracker_params_path, "r") as f:
            cfg = yaml.safe_load(f)
        ds_cfg = cfg.get("deepsort", {})

        # Populate args
        self.args = DeepSortTrackerArgs(
            metric=ds_cfg.get("metric", "cosine"),
            matching_threshold=ds_cfg.get("matching_threshold", 0.2),
            budget=ds_cfg.get("budget", None),
            max_iou_distance=ds_cfg.get("max_iou_distance", 0.7),
            max_age=ds_cfg.get("max_age", 30),
            n_init=ds_cfg.get("n_init", 3),
            det_conf=ds_cfg.get("det_conf", 0.5),
        )

        # Build metric and tracker
        self.metric = NearestNeighborDistanceMetric(
            self.args.metric,
            self.args.matching_threshold,
            self.args.budget
        )
        self.tracker = DeepSortTrackerClass(
            metric=self.metric,
            max_iou_distance=self.args.max_iou_distance,
            max_age=self.args.max_age,
            n_init=self.args.n_init
        )

        self.reid_model = reid_model  # optional appearance model
        self.frame_id = 0
        self.track_classes: Dict[int, int] = {}
        self.track_scores: Dict[int, float] = {}

    def update(self, detections: List, frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        :param detections: list of detection objects (x1,y1,x2,y2,confidence[,class_id])
        :param frame: HxWxC image array for appearance cropping
        :returns: list of Track(track_id,x1,y1,x2,y2,score,class_id)
        """
        self.frame_id += 1

        # 1) Filter by detection confidence
        dets = [d for d in detections if d.confidence > self.args.det_conf]
        if not dets:
            # advance internal time and clear short-lived tracks
            self.tracker.predict()
            self.tracker.update([])
            return []
        
        det_bboxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
        det_classes = [d.class_id for d in dets]
        det_scores = [d.confidence for d in dets]

        # 2) Gather bboxes (tlwh format) and scores
        # DeepSort Detection expects boxes as [top, left, width, height]
        bboxes_tlwh = np.array([
            [d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1]
            for d in dets
        ], dtype=np.float32)
        scores = np.array([d.confidence for d in dets], dtype=float)

        # 3) Compute appearance features if model provided
        if self.reid_model is not None and frame is not None:
            features = []
            for d in dets:
                x1,y1,x2,y2 = map(int,(d.x1,d.y1,d.x2,d.y2))
                crop = frame[y1:y2, x1:x2]
                features.append(self.reid_model(crop))
        else:
            features = [np.ones((1,),dtype=float) for _ in dets]
        ds_detections = [Detection(tlwh, score, feat)
                         for tlwh, score, feat in zip(bboxes_tlwh, det_scores, features)]
        # run tracker
        self.tracker.predict()
        self.tracker.update(ds_detections)
        # IoU helper
        def iou(a,b):
            xa1,ya1,xa2,ya2 = a
            xb1,yb1,xb2,yb2 = b
            xi1, yi1 = max(xa1,xb1), max(ya1,yb1)
            xi2, yi2 = min(xa2,xb2), min(ya2,yb2)
            inter = max(0, xi2-xi1)*max(0, yi2-yi1)
            union = (xa2-xa1)*(ya2-ya1)+(xb2-xb1)*(yb2-yb1)-inter+1e-6
            return inter/union
        # build output
        out_tracks: List[Track] = []
        for trk in self.tracker.tracks:
            if not trk.is_confirmed():
                continue
            # fetch bbox
            tlwh = trk.to_tlwh()
            x1,y1,w,h = tlwh; x2,y2 = x1+w,y1+h
            bbox = np.array([x1,y1,x2,y2],dtype=np.float32)
            tid = int(trk.track_id)
            # match to original detection
            if len(dets)>0:
                ious = [iou(bbox,db) for db in det_bboxes]
                bid = int(np.argmax(ious))
                if ious[bid] > 0.5:
                    cls = det_classes[bid]
                    sc  = det_scores[bid]
                else:
                    cls = self.track_classes.get(tid,0)
                    sc  = self.track_scores.get(tid,1.0)
            else:
                cls = self.track_classes.get(tid,0)
                sc  = self.track_scores.get(tid,1.0)
            # store
            self.track_classes[tid]=cls
            self.track_scores[tid]=sc
            out_tracks.append(Track(
                track_id=tid,
                x1=float(x1),y1=float(y1),x2=float(x2),y2=float(y2),
                score=float(sc),
                class_id=int(cls)
            ))
        return out_tracks

    def reset(self):
        """Clear the tracker state and ID counter."""
        self.frame_id = 0
        self.tracker.tracks = []
        self.tracker._next_id = 1

