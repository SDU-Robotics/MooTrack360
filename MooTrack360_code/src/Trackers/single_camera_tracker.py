import cv2
import numpy as np
import yaml
from scipy.interpolate import splprep, splev
import time
import logging

from Cam_models.dscamera.camera import DSCamera

logging.basicConfig(filename='latency_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')


class SingleCameraTracker:
    def __init__(self, setup_config, detector, tracker, cam_calib: str = "VelKoTek/src/configs/cam_calibration.json"):
        
        self.mode = setup_config["mode"]
        if self.mode =="SCT":
            self.stable = setup_config["stable"]
            self.area = setup_config["area"]

            first_cam_params = next(iter(setup_config["cameras"].values()))
            self.camera_params = first_cam_params
            self.camera = DSCamera(json_filename=cam_calib)
            
            # The mask or ROI to specify the area of interest
            self.roi = {}
            
            roi_points = self.camera_params.get("mask_roi", [])

            if roi_points:
                roi_points_tuples = [tuple(pt) for pt in roi_points]
                self.roi[self.camera_params["id"]] = roi_points_tuples

        elif self.mode =="MCT":
            # Needs to handle how multiple cameras are initiated
            pass

        self.detector = detector
        self.tracker = tracker

        self.local_tracklets = []
        # e.g., initialize a SORT or DeepSORT instance

    def process_frame(self, frame, poseEstimate:bool = True) -> list[dict]:
        """
        Run detection + tracking on a single frame.
        Returns a list of track objects with IDs and bounding boxes.
        """
        # 1. preprocessing step (ROI_masking)
        # 2. detector.detect_objects(frame)
        # 3. pass detections to tracker
        # 4. return tracklets

        # Apply the detector        
        detections = self.detector.detect_objects(frame)

        # Time the tracker update
        start_time = time.time()
        out_tracks = self.tracker.update(detections, frame)
        end_time = time.time()
        
        # Calculate latency
        latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds

        # Log the latency
        logging.info(f"Tracker update latency: {latency_ms:.2f} ms")
        
        tracklets = []
        for t in out_tracks:
            track_dict = {
                "track_id": t.track_id,
                "bbox": [t.x1, t.y1, t.x2, t.y2],
                "score": t.score,
                "class_id": t.class_id
            }
            tracklets.append(track_dict)

        # (Optional) store for local reference
        # self.local_tracklets = tracklets

        return tracklets

    def run_stream(self):
        """
        Continuously read from camera, process frames,
        and emit results (maybe to a queue or callback).
        """
        pass
