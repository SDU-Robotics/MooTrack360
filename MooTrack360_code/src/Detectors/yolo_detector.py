import numpy as np
import yaml
from typing import List
from ultralytics import YOLO
import torch

from .base_detector import BaseDetector, Detection
from pathlib import Path

class YOLOvDetector (BaseDetector):
    """
    A concrete detector that uses the Ultralytics YOLO model (YOLOv8).
    """

    def __init__(self, model_path: str, device: str = "cuda", predict_params_path: str = str(Path("VelKoTek","src","configs","detector_configs.yaml"))):
        """
        :param model_path: Path to the Ultralytics YOLO model (.pt file), e.g. 'yolov8n.pt'
        :param device: 'cpu' or 'cuda'
        """
        try:
            super().__init__()

            # Compute the absolute path for the model file relative to this file
            base_dir = Path(__file__).resolve().parent  # Directory of yolo_detector.py
            abs_model_path = (base_dir / model_path).resolve()

            # Print the resolved path for debugging
            print("Loading model from:", abs_model_path)

            # Initialize the YOLO model using the absolute path
            self.model = YOLO(str(abs_model_path),task="detect")

            # if device == "cuda" and torch.cuda.is_available():
            #     self.model.to("cuda")
            #     self.device = "cuda"
            # else:
            #     self.model.to("cpu")
            #     self.device = "cpu"

            # 3. Load predict params from YAML
            with open(predict_params_path, "r") as f:
                config = yaml.safe_load(f)

            # 4. Store the predict parameters for later use
            self.predict_params = config["YOLO_predict"]

        except Exception as e:
            # Wrap any exception in a RuntimeError to provide context
            raise RuntimeError(f"Failed to initialize YOLOvDetector: {e}")


    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs the Ultralytics YOLO model on the given frame and returns a list of detections.

        :param frame: A NumPy array (H x W x 3). Typically BGR if read by OpenCV.
        :return: List of Detection objects
        """
        # If your model is trained for RGB images but your frame is BGR (OpenCV default),
        # you may want to convert: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference; 'predict' can accept np.ndarray or file paths
        results = self.model.predict(source=frame, **self.predict_params)  # conf=0.25 as an example

        # results is typically a list of 'Boxes' objects (one for each image).
        # Here we assume we are passing only one image at a time, so we take results[0].
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes  # Ultralytics 'Boxes' object
            for box in boxes:
                # box.xyxy, box.conf, and box.cls are Tensors
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Create our standardized Detection object
                detection = Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    class_id=cls
                )
                detections.append(detection)

        return detections