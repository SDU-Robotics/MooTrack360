import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import queue
import yaml
import cv2
from vidgear.gears import VideoGear

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
from Trackers.single_camera_tracker import SingleCameraTracker
from Storage.database import Session
from Storage.cow_instance import Detection
from Visualizers.instance_viz import visualize_with_supervision

from Processors.processors import VideoCaptureThread, frame_processor, db_writer_worker, visualize_and_save


########################################################################
# Shared Memory + Ring Buffer Constants
########################################################################
RING_BUFFER_SIZE = 10  # Number of frames in the shared buffer
CHANNELS = 3           # Typically 3 for BGR or RGB

#main_lock = threading.Lock()

########################################################################
# Main Execution
########################################################################
def main():
    # 1) Load config
    config = load_config(
        mode="SCT",
        main_config_path=str(Path("VelKoTek", "src", "configs", "execute_configs.yaml")),
        env_config_path=str(Path("VelKoTek", "src", "configs", "stable_configs.yaml"))
    )

    # 2) Determine video properties from first frame
    video_path = config["videos"][0]
    cap = cv2.VideoCapture(video_path)
    config["fps"] = int(cap.get(cv2.CAP_PROP_FPS))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_resize = config["img_resize"]
    if img_resize != 1.0:
        config["width"] = int(width * img_resize)
        config["height"] = int(height * img_resize)
    else:
        config["width"] = width
        config["height"] = height

    print(f"Original Video => FPS: {config['fps']}, Width: {width}, Height: {height}")
    print(f"Using resized => Width: {config['width']}, Height: {config['height']}")

    ret, first_frame = cap.read()
    if not ret:
        print("Could not read a frame from video. Exiting.")
        cap.release()
        return

    # Resize first_frame if needed
    if img_resize != 1.0:
        frame_ori = cv2.resize(
            first_frame,
            (config["width"], config["height"]),
            interpolation=cv2.INTER_AREA,
        )
    else:
        frame_ori = first_frame

    # We need to figure out the shape of the masked frame
    camera_params, roi = mask_params(config)
    masked_frame, topLeftCrop = apply_masking(frame_ori, camera_params, roi)

    # Now we know the shape for the original buffer and masked buffer
    shape_ori = (RING_BUFFER_SIZE, config["height"], config["width"], CHANNELS)
    shape_masked = (
        RING_BUFFER_SIZE,
        masked_frame.shape[0],
        masked_frame.shape[1],
        CHANNELS
    )
    dtype = np.uint8

    # We'll re-open the video from the beginning for the capture thread
    cap.release()

    # 3) Create shared memory blocks
    #    -> One for the original frames, one for the masked frames
    shared_mem_size_ori = np.prod(shape_ori) * np.dtype(dtype).itemsize
    shm_ori = SharedMemory(create=True, size=shared_mem_size_ori)
    ring_buffer_np_ori = np.ndarray(shape_ori, dtype=dtype, buffer=shm_ori.buf)

    shared_mem_size_masked = np.prod(shape_masked) * np.dtype(dtype).itemsize
    shm_masked = SharedMemory(create=True, size=shared_mem_size_masked)
    ring_buffer_np_masked = np.ndarray(shape_masked, dtype=dtype, buffer=shm_masked.buf)

    # 4) Create multiprocessing queues
    available_indices_queue = mp.Queue(maxsize=RING_BUFFER_SIZE)
    frame_index_queue = mp.Queue(maxsize=RING_BUFFER_SIZE)
    result_queue = mp.Queue(maxsize=RING_BUFFER_SIZE)
    db_write_queue = mp.Queue(maxsize=10000)

    # Initialize ring-buffer slots
    for i in range(RING_BUFFER_SIZE):
        available_indices_queue.put(i)

    # 5) Optionally start DB writer thread
    if config["save_database"]:
        db_writer_thread = threading.Thread(
            target=db_writer_worker,
            kwargs={'db_write_queue': db_write_queue, 'batch_size': 300, 'timeout': 1},
            daemon=True
        )
        db_writer_thread.start()

    # 6) Start child process for detection/tracking
    processing_process = mp.Process(
        target=frame_processor,
        args=(
            shm_ori.name,
            shape_ori,
            shm_masked.name,
            shape_masked,
            dtype,
            frame_index_queue,
            result_queue,
            db_write_queue,
            config,
        ),
        daemon=True
    )
    processing_process.start()

    # 7) Start capture thread in main process
    # Re-open the video from the beginning for the capture thread:
    cap = None  # not strictly needed, just clarifies we closed it
    video_thread = VideoCaptureThread(
        video_path,
        ring_buffer_np_ori,
        ring_buffer_np_masked,
        available_indices_queue,
        frame_index_queue,
        config
    )
    video_thread.start()

    # 8) Visualization thread
    if config["viz"] or config["save_video"]:
        visualization_thread = threading.Thread(
            target=visualize_and_save,
            args=(result_queue, ring_buffer_np_ori, available_indices_queue, config),
            daemon=True
        )
        time.sleep(1)
        visualization_thread.start()

        # Wait for capture to finish
        video_thread.join()
        processing_process.join()
        visualization_thread.join()
    else:
        # If no visualization, just wait
        video_thread.join()
        processing_process.join()

    # 9) DB thread shutdown
    if config["save_database"]:
        db_write_queue.put(None)
        db_writer_thread.join()

    # 10) Cleanup shared memory
    shm_ori.close()
    shm_ori.unlink()
    shm_masked.close()
    shm_masked.unlink()

    print("Finished Processing")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
