import yaml
import torch
import time
import psutil
import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from collections import deque
from scipy.interpolate import splprep, splev
from Detectors.base_detector import Detection
from Trackers.single_camera_tracker import SingleCameraTracker

# Mapping for class names
CLASS_NAME_MAP = {
    0: "laying",
    1: "standing"
}

def visualize_with_supervision(frame, tracklets: list[dict], camera_params, roi, frame_num: int, start_time, frameShift) -> np.ndarray:
    """
    Convert custom tracklets (each a dict with keys "track_id", "bbox", "score", "class_id")
    into Supervision's Detections and use annotators to draw bounding boxes and/or labels onto the frame.
    "frameShift": the topleft of the masked image, which is cropped from "frame", before passed through detection and tracking
    
    The label text is built based on the visualization configuration:
      - If "class" is True, the class name is included.
      - The track ID is always included.
      - If "score" is True, the confidence score is appended.
    """
    # Load visualization configuration from YAML file
    with open(str(Path("VelKoTek","src","configs","visualization_config.yaml")), "r") as config_file:
        vis_config = yaml.safe_load(config_file)

    # Use default values if any key is missing.
    INSTANCE_VIZ_CONFIG = vis_config.get("instance_viz", {
        "box": True, 
        "class": True,
        "id": True, 
        "score": True,
        "roi": True
        })

    # Return the original frame if there are no tracklets.
    if not tracklets:
        return frame

    # 1) Build an array for bounding boxes from each tracklet's "bbox" field.
    # (bbox: topLeft_x, topLeft_y, bottomRight_x, bottomRight_y)
    detection_data = np.array(
        [[track["bbox"][0]+ frameShift[0], track["bbox"][1]+ frameShift[1], track["bbox"][2]+ frameShift[0], track["bbox"][3]+ frameShift[1]] for track in tracklets],
        dtype=float
    )

    # Ensure the array has shape (n, 4); if it's empty, force shape (0, 4).
    if detection_data.ndim == 1:
        detection_data = np.empty((0, 4), dtype=float)
    
    # 2) Build an array for the confidence scores.
    scores = np.array(
        [track["score"] for track in tracklets],
        dtype=float
    )
    
    # 3) Build an array for track IDs.
    track_ids = np.array(
        [track["track_id"] for track in tracklets],
        dtype=int
    )
    
    # 4) Get raw class IDs from the tracklets; default to -1 if not provided.
    raw_class_ids = [track.get("class_id", -1) for track in tracklets]
    # Create a safe version (all negative IDs become 0) for the Detections object.
    safe_class_ids = np.array([cid if cid >= 0 else 0 for cid in raw_class_ids], dtype=int)
    
    # 5) Prepare labels based on configuration.
    labels = []
    for cid, tid, sc in zip(raw_class_ids, track_ids, scores):
        label_parts = []
        if INSTANCE_VIZ_CONFIG.get("class", True):
            class_name = CLASS_NAME_MAP.get(cid, "unknown") if cid >= 0 else "unknown"
            label_parts.append(f"Class: {class_name}")
        if INSTANCE_VIZ_CONFIG.get("id", True):
            label_parts.append(f"ID: {tid}")
        if INSTANCE_VIZ_CONFIG.get("score", True):
            label_parts.append(f"{sc:.2f}")
        labels.append(", ".join(label_parts))
    
    # 6) Create a Supervision Detections object.
    sv_detections = sv.Detections(
        xyxy=detection_data,
        confidence=scores,
        class_id=safe_class_ids  # Use safe class IDs to avoid negative indices.
    )
    
    # 7) Annotate the frame based on configuration.
    annotated_frame = frame.copy()

    if INSTANCE_VIZ_CONFIG.get("roi", True):
        annotated_frame = apply_masking(annotated_frame, roi, camera_params)
    
    # Draw bounding boxes if enabled.
    if INSTANCE_VIZ_CONFIG.get("box", True):
        box_annotator = sv.BoxAnnotator(
            thickness=1,
            #text_thickness=1,
            #text_scale=0.3,     
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=sv_detections,
            #labels=labels
        )
    
    # Draw labels if any of class, id, or score is enabled.
    if (INSTANCE_VIZ_CONFIG.get("class", True) or 
        INSTANCE_VIZ_CONFIG.get("id", True) or 
        INSTANCE_VIZ_CONFIG.get("score", True)):
        label_annotator = sv.LabelAnnotator(
            text_thickness=1,
            text_scale=0.3,
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=sv_detections,
            labels=labels
        )
    if INSTANCE_VIZ_CONFIG.get("FPS", True):
        avg_fps, latency = track_realtime_performance(frame_num, start_time)

        # Display FPS on frame
        #cv2.putText(annotated_frame, f"FPS: {avg_fps:.2f}", (10, 40),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return annotated_frame

def apply_masking(frame, roi, camera_params):
    """
    Apply the defined mask from the config file to define
    the region-of-interest (ROI)
    """

    # 1) Get image dimensions
    h, w = frame.shape[:2]

    # 2) Denormalize control points to image pixel coordinates
    denorm_points = [(int(x_norm * w), int(y_norm * h))
                    for (x_norm, y_norm) in roi[camera_params["id"]]]
    
    # 3) Compute the B-spline curve in pixel coordinates
    curve_pts = b_spline_curve(denorm_points)

    # Reshape for use with cv2.polylines (expects an array of shape [num_points, 1, 2])
    polygon_points = curve_pts.reshape((-1, 1, 2))
    
    # 4) Draw the ROI polygon on the image
    #    isClosed=True ensures that the polygon is closed.
    #    Change the color and thickness as needed.
    cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

    return frame

def b_spline_curve(control_points, num_points=200, smoothing=0.0, closed=True):
    """
    Computes a B-spline curve for the given control points using SciPy's splprep/splev.
    
    control_points : list of (x, y) in pixel coordinates
    num_points     : how many interpolated points to produce
    smoothing      : smoothing factor for the B-spline
    closed         : if True, make the B-spline periodic (closed shape)
    
    Returns np.array of shape (num_points, 2).
    """
    if len(control_points) < 2:
        return np.array(control_points, dtype=np.int32)
    
    points_arr = np.array(control_points, dtype=np.float32)
    x = points_arr[:, 0]
    y = points_arr[:, 1]

    try:
        tck, _ = splprep([x, y], s=smoothing, per=closed)
        u_fine = np.linspace(0, 1, num_points)
        x_fine, y_fine = splev(u_fine, tck)
        curve = np.vstack([x_fine, y_fine]).T
        return curve.astype(np.int32)
    except Exception as e:
        print("Error in B-spline computation:", e)
        # fallback: return original points if something fails
        return points_arr.astype(np.int32)



# Store past frame times to calculate smooth FPS
frame_times = deque(maxlen=30)  # Store the last 30 frame times

def track_realtime_performance(frame_num, start_time):
    """Logs FPS, processing time, CPU, Memory, and GPU usage."""
    process = psutil.Process()
    cpu_percent = process.cpu_percent(interval=None)
    memory_info = process.memory_info().rss / (1024 * 1024)
    gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

    # Compute processing time and FPS
    end_time = time.time()
    processing_time = end_time - start_time
    frame_times.append(processing_time)

    # Calculate rolling FPS (avoid division by zero)
    avg_fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0
    latency = processing_time

    #print(f"[Frame {frame_num}] FPS: {avg_fps:.2f}, Latency: {latency:.4f}s, CPU: {cpu_percent:.2f}%, "
    #      f"Memory: {memory_info:.2f}MB, GPU: {gpu_mem:.2f}MB")

    return avg_fps, latency