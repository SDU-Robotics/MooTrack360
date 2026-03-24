import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import queue
import yaml
import cv2
from vidgear.gears import VideoGear
import cProfile
import pstats
import os
import csv

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from typing import List, Dict, Any
from collections import defaultdict
import threading
from datetime import datetime
from collections import deque
from pathlib import Path

# Local imports (replace with your real modules)
from utils.utils import load_config, apply_masking, mask_params
from Detectors.yolo_detector import YOLOvDetector
from Detectors.coco_detector import COCODetector
from Detectors.rtdetr_detector import RTDETRDetector
from Trackers.bytetrack_tracker import ByteTrackTracker
from Trackers.botsorttrack_tracker import BotSortTracker
from Trackers.smiletrack_tracker import SMILETrackTracker
from Trackers.sort_tracker import SortTracker
from Trackers.ocsort_tracker import OCSortTracker
from Trackers.tracktor_tracker import TracktorTracker
from Trackers.deepsort_tracker import DeepSortTracker
from Trackers.conftrack_tracker import ConfTrackTracker
from Trackers.gentrack_tracker import GenTrackTracker
from Trackers.single_camera_tracker import SingleCameraTracker
from Storage.database import Session
from Storage.cow_instance import Detection
from Visualizers.instance_viz import visualize_with_supervision

def shift_bbox(bbox, shift):
    dx, dy = shift
    x1, y1, x2, y2 = bbox
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

def mot_row(frame_idx, track, shift):
    """Return a 10-column MOT row (as list) for one tracklet box."""
    x1, y1, x2, y2 = shift_bbox(track["bbox"], shift)
    w, h = x2 - x1, y2 - y1
    return [
        frame_idx,                 # col-1  frame (1-based)
        track["track_id"],         # col-2  id
        int(x1), int(y1),          # col-3 4
        int(w),  int(h),           # col-5 6
        track.get("score", 1),     # col-7  detection / tracker score
        track.get("class_id", -1), # col-8  class
        -1,                        # col-9  visibility   (not used)
        -1                         # col-10 x,y,z world (2-D → -1)
    ]

#process_lock = threading.Lock()

########################################################################
# Database Writer Thread
########################################################################
def db_writer_worker(db_write_queue:mp.Queue, batch_size=300, timeout=1):
    # profiler = cProfile.Profile()
    # profiler.enable()
    session = Session()  # Dedicated DB session
    batch = []
    while True:
        previous_time = time.time()
        #print("db_write_queue", db_write_queue.qsize())
        #print("batch", len(batch))
        try:
            item = db_write_queue.get(timeout=timeout)
            if item is None:
                # Termination signal: commit remaining batch and exit
                if batch:
                    try:
                        session.add_all(batch)
                        session.commit()
                        batch = []
                    except Exception as e:
                        session.rollback()
                        #print("DB write error during final batch:", e)
                break
            batch.append(item)
            if len(batch) >= batch_size:
                try:
                    session.add_all(batch)
                    session.commit()
                    batch = []
                except Exception as e:
                    session.rollback()
                    #print("DB write error:", e)
                    batch = []
        except queue.Empty:
            if batch:
                try:
                    session.add_all(batch)
                    session.commit()
                    batch = []
                except Exception as e:
                    session.rollback()
                    #print("DB write error (empty timeout):", e)
                    batch = []
        current_time = time.time()
        #print("\nDB Writer Time:", (current_time - previous_time))
    session.close()
    # profiler.disable()
    # # Dump stats to a file named after the process (or slot_idx, or anything unique)
    # profiler.dump_stats(f"child_process_{os.getpid()}.prof")

########################################################################
# Video Capture Thread
########################################################################
class VideoCaptureThread(threading.Thread):
    """
    Continuously captures frames from the video source and writes them
    into two shared-memory ring buffers:
      - ring_buffer_np_ori   -> for the original (resized) frame
      - ring_buffer_np_masked-> for the masked/cropped frame
    """
    def __init__(
        self,
        video_path: str,
        ring_buffer_np_ori: np.ndarray,
        ring_buffer_np_masked: np.ndarray,
        available_indices_queue: mp.Queue,
        frame_index_queue: mp.Queue,
        config: dict,
    ):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.ring_buffer_np_ori = ring_buffer_np_ori
        self.ring_buffer_np_masked = ring_buffer_np_masked
        self.available_indices_queue = available_indices_queue
        self.frame_index_queue = frame_index_queue

        self.stream = VideoGear(source=video_path).start()
        self.running = True

        self.camera_params, self.roi = mask_params(config)
        self.config = config
        self.img_resize = config["img_resize"]        

    def run(self):
        # profiler = cProfile.Profile()
        # profiler.enable()
        while self.running:
            previous_time = time.time()
            frame = self.stream.read()
            if frame is None:
                # End of stream
                break

            # Resize if needed
            if self.img_resize != 1.0:
                frame_ori = cv2.resize(
                    frame,
                    (self.config["width"], self.config["height"]),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                frame_ori = frame

            # Create masked (cropped) version
            masked_frame, topLeftCrop = apply_masking(
                frame_ori, self.camera_params, self.roi
            )

            # Wait for an available ring-buffer slot
            try:
                slot_idx = self.available_indices_queue.get(timeout=1)
            except queue.Empty:
                # If no slots available, optionally drop the frame
                #print("No available ring-buffer slot, dropping frame.")
                continue

            # Write the original frame and the masked frame into separate buffers
            self.ring_buffer_np_ori[slot_idx, :, :, :] = frame_ori
            self.ring_buffer_np_masked[slot_idx, :, :, :] = masked_frame

            # Send the slot index + topLeftCrop to the child process
            # to indicate a new frame is ready.
            self.frame_index_queue.put((slot_idx, topLeftCrop))

            current_time = time.time()
            #print("Video Capture Time", (current_time - previous_time))

        # Stream is finished
        self.stream.stop()
        # Signal the child process no more frames
        self.frame_index_queue.put((-1, None))
        # profiler.disable()
        # # Dump stats to a file named after the process (or slot_idx, or anything unique)
        # profiler.dump_stats(f"child_process_{os.getpid()}.prof")

########################################################################
# Child Process: Frame Processor
########################################################################
def frame_processor(
    shm_name_ori: str,
    shape_ori: tuple,
    shm_name_masked: str,
    shape_masked: tuple,
    dtype: np.dtype,
    frame_index_queue: mp.Queue,
    result_queue: mp.Queue,
    db_write_queue: mp.Queue,
    config: dict
):
    """
    Reads frames from the two shared-memory ring buffers (original + masked),
    performs detection/tracking on the masked frame,
    then sends results (slot_idx + tracklets + metadata) back to the main process.
    """
    # Attach to the existing shared memory regions for both ori and masked
    existing_shm_ori = SharedMemory(name=shm_name_ori)
    ring_buffer_np_ori = np.ndarray(shape_ori, dtype=dtype, buffer=existing_shm_ori.buf)

    existing_shm_masked = SharedMemory(name=shm_name_masked)
    ring_buffer_np_masked = np.ndarray(shape_masked, dtype=dtype, buffer=existing_shm_masked.buf)

    # Load your detector in the child process
    detector_cfg = config["detector"]
    if detector_cfg["type"] == "yolo":
        detector = YOLOvDetector(
            model_path=detector_cfg["model_path"], device=detector_cfg["device"]
        )
    elif detector_cfg["type"] == "rtdetr":
        detector = RTDETRDetector(
            model_path=detector_cfg["model_path"], device=detector_cfg["device"]
        )
    elif detector_cfg["type"] == "coco":
        detector = COCODetector(
            model_path=detector_cfg["model_path"], device=detector_cfg["device"]
        )

    # Load tracker
    tracker_cfg = config["tracker"]
    if tracker_cfg["type"] == "bytetracker":
        tracker = ByteTrackTracker()
    elif tracker_cfg["type"] == "botsort":
        tracker = BotSortTracker()
    elif tracker_cfg["type"] == "smiletracker":
        tracker = SMILETrackTracker()
    elif tracker_cfg["type"] == "ocsort":
        tracker = OCSortTracker()
    elif tracker_cfg["type"] == "sort":
        tracker = SortTracker()
    elif tracker_cfg["type"] =="deepsort":
        tracker = DeepSortTracker()
    elif tracker_cfg["type"] == "tracktor":
        tracker = TracktorTracker()
    elif tracker_cfg["type"] == "conftrack":
        tracker = ConfTrackTracker()
    elif tracker_cfg["type"] == "gentrack":
        tracker = GenTrackTracker()
        
    # Create SingleCameraTracker in child
    single_cam_tracker = SingleCameraTracker(config, detector, tracker)
    camera_params = single_cam_tracker.camera_params
    roi = single_cam_tracker.roi
    
    # profiler = cProfile.Profile()
    # profiler.enable()
    frame_idx = 0  # Initialize frame_idx

    while True:
        previous_time = time.time()
        
        data = frame_index_queue.get()
        if data is None:
            continue
        slot_idx, topLeftCrop = data

        if slot_idx < 0:
            # No more frames
            break

        # Grab the masked frame (to run detection/tracking on)
        frame_masked = ring_buffer_np_masked[slot_idx].copy()

        # If you ever need the original frame for some reason, you can do:
        # frame_ori = ring_buffer_np_ori[slot_idx].copy()

        # Run detection/tracking on the masked frame
        tracklets = single_cam_tracker.process_frame(frame=frame_masked)
        processed_time = time.time()

        # Optionally queue detection objects for DB
        if config.get("save_database", False):
            for track in tracklets:
                track_id = track.get("track_id")
                state = "Standing" if track.get("class_id") == 1 else "Lying"
                detection = Detection(
                    tracklet_id=track_id,
                    timestamp=datetime.utcnow(),
                    x1=track["bbox"][0],
                    y1=track["bbox"][1],
                    x2=track["bbox"][2],
                    y2=track["bbox"][3],
                    posture=state
                )
                db_write_queue.put(detection)


        # Send results back to main
        if result_queue.full():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass

        # The main process uses slot_idx to retrieve the "original" frame for visualization
        # We also pass topLeftCrop, processed_time, etc.
        frame_idx += 1  
        result_queue.put((slot_idx, tracklets, topLeftCrop, processed_time, frame_idx))
        current_time = time.time()
        #print("\nFrame Process Time", (current_time - previous_time))

    # Cleanup
    existing_shm_ori.close()
    existing_shm_masked.close()
    
    # profiler.disable()
    # # Dump stats to a file named after the process (or slot_idx, or anything unique)
    # profiler.dump_stats(f"child_process_{os.getpid()}.prof")


########################################################################
# Visualization and Video Writer Thread
########################################################################
def visualize_and_save(
    result_queue: mp.Queue,
    ring_buffer_np_ori: np.ndarray,
    available_indices_queue: mp.Queue,
    config: dict
):
    """
    Continuously reads new results (slot index + tracklets), grabs the
    corresponding *original* frame from shared memory for visualization,
    then returns that slot index to the pool of available indices.
    """
    if config["trackeval"]:
        bench      = config.get("te_benchmark", "COWS")      # e.g. COWS
        split      = config.get("te_split",     "train")     # train / val
        tracker_nm = config["tracker"]["type"] # your algo
        sequence   = Path(config["videos"][0]).stem   # video name

        te_dir = Path("VelKoTek", "implementation_outputs",config["output_path"], "data",)
        te_dir.mkdir(parents=True, exist_ok=True)
        mot_rows = []            # collect rows in RAM first
        frame_idx = 0  

    if config["save_video"]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video_path = str(
            Path("VelKoTek", "implementation_outputs", config["output_path"], "annotated_output.mp4")
        )
        out = cv2.VideoWriter(
            out_video_path, fourcc, config["fps"], (config["width"], config["height"])
        )

    # Rolling window for FPS calculation
    timestamp_window = deque(maxlen=30)

    camera_params, roi = mask_params(config)

    # profiler = cProfile.Profile()
    # profiler.enable()

    while True:
        previous_time = time.time()
        try:
            # Wait up to 1 second for a new processed result
            result = result_queue.get(timeout=1)
        except queue.Empty:
            #print("No new frames received for 1 second. Exiting visualization.")
            break

        if result is None:
            #print("Termination signal in visualization.")
            break

        # Drain queue for the latest
        while True:
            try:
                latest = result_queue.get_nowait()
                result = latest
            except queue.Empty:
                break

        slot_idx, tracklets, topLeftCrop, processed_time, frame_idx = result
        timestamp_window.append(processed_time)

        # Calculate approximate FPS
        if len(timestamp_window) >= 2:
            fps = len(timestamp_window) / (timestamp_window[-1] - timestamp_window[0])
        else:
            fps = 0.0

        # Get the original (resized) frame for annotation
        frame_ori = ring_buffer_np_ori[slot_idx].copy()

        annotated_frame = visualize_with_supervision(
            frame_ori, tracklets, camera_params, roi, len(timestamp_window), processed_time, topLeftCrop
        )
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (int(config["width"] * 0.75), 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if config["viz"]:
            cv2.imshow("Tracking", annotated_frame)
            cv2.waitKey(1)

        if config["save_video"]:
            out.write(annotated_frame)

        if config["trackeval"]:
            for trk in tracklets:
                mot_rows.append(mot_row(frame_idx, trk, topLeftCrop))

        # Return slot to available
        available_indices_queue.put(slot_idx)
        current_time = time.time()
        #print("\nVisualization Time", (current_time - previous_time))
        #print("\n======================================")

    # profiler.disable()
    # # Dump stats to a file named after the process (or slot_idx, or anything unique)
    # profiler.dump_stats(f"child_process_{os.getpid()}.prof")
    if config["save_video"]:
        out.release()
    cv2.destroyAllWindows()

    if config["trackeval"]:
        out_file = te_dir / f"{sequence}.txt"
        # sort by frame then id (TrackEval requirement)
        mot_rows.sort(key=lambda r: (r[0], r[1]))
        with open(out_file, "w", newline="") as f:
            csv.writer(f).writerows(mot_rows)
        print(f"[✓] Tracker result saved for TrackEval → {out_file}")