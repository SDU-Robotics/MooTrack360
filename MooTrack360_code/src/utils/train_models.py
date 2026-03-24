import os
from ultralytics import YOLO
from ultralytics import RTDETR
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import cv2
import time

import shutil

torch.hub.set_dir('VelKoTek/src/Detectors/detection_models/model_cache')

MODEL_CACHE_DIR = "VelKoTek/src/Detectors/detection_models/model_cache/"

""" 
This script is used to train multiple detection models at once and output results in the
"training_outputs" folder. The nameing of the resulting folder is the day and timeslot the script was run.

Format: run_YYYY-MM-DD_HH_MM_SS (H = Hour, M =Minutes and S = second)

The script is able to train YOLO models. To change parameters etc modify the following functions: 
- YOLO_trainer()

Epochs and image size can be specified as well as the dataset in use. The path to the .yaml file (for YOLO format) or dataset path (for COCO format).
As well as a test image needs to be specified.

Hyperparameters needs to be changes in the two main functions:
- Train_YOLO_models()

The other functions acts as helper functions:
- log_training_details
- plot_performance_graph_YOLO()

Note: The focus has been on the ultralytics yolo models, and therefor the COCO has not been validated yet.

"""

def log_training_details(output_dir, model_name, config, data_path, epochs, imgsz, metrics):
    """Save training configuration and results in a text file."""
    log_path = os.path.join(output_dir, "training_details.txt")
    with open(log_path, "w") as f:
        f.write("Training Configuration and Results\n")
        f.write("="*40 + "\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Model Path: {config['model_path']}\n")
        f.write(f"Data Path: {data_path}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Image Size: {imgsz}\n")
        f.write("\nMetrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def plot_performance_graph_YOLO(performance_metrics, output_base_dir):
    """Plot detection metrics and a separate bar chart for inference time."""
    names = performance_metrics["model_name"]
    # --- Top: line plots for detection scores ---
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 10),
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Line plots
    ax1.plot(names, performance_metrics["mAP@0.5"],      marker='o', label="mAP@0.5")
    ax1.plot(names, performance_metrics["mAP@[0.5:0.95]"], marker='o', label="mAP@[0.5:0.95]")
    ax1.plot(names, performance_metrics["Precision"],     marker='o', label="Precision")
    ax1.plot(names, performance_metrics["F1"],            marker='o', label="F1")

    ax1.set_title("YOLO Model Detection Metrics")
    ax1.set_ylabel("Score")
    ax1.set_xticks(names)
    ax1.set_xticklabels(names, rotation=45)
    ax1.legend()
    ax1.grid(True)

    # --- Bottom: bar chart for inference time ---
    ax2.bar(names, performance_metrics["Time"])
    ax2.set_title("Average Inference Time per Image")
    ax2.set_ylabel("Time (ms)")
    ax2.set_xticks(names)
    ax2.set_xticklabels(names, rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(output_base_dir, "performance_metrics.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Performance metrics plot saved at {plot_path}")

def Train_YOLO_models(model_configs, data_path, image_path, output_base_dir, epochs=100, imgsz=640, format="engine", dynamic=False):
    # Create output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Generate a timestamp for the current run
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    timestamped_dir = os.path.join(output_base_dir, f"run_{current_time}")
    os.makedirs(timestamped_dir, exist_ok=True)
    print(f"Created timestamped directory: {timestamped_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Dictionary to store performance metrics for plotting
    performance_metrics = {    
        "model_name": [],
        "mAP@0.5": [],
        "mAP@[0.5:0.95]": [],
        "Recall": [],
        "Precision": [],
        "F1": [],
        "Time": []
    }

    # Iterate through each model configuration
    for config in model_configs:
        try:
            model_name = config["name"]
            model_path = MODEL_CACHE_DIR+config["model_path"]

            # Create an output directory for the current model inside the timestamped directory
            model_output_dir = os.path.join(timestamped_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            print(f"Training {model_name} model...")

            # Fix relative path ultralytics bug
            data_path_resolved = os.path.relpath(data_path, start=os.getcwd())
            print(data_path_resolved)

            if model_name == 'rtdetr-l' or model_name == 'rtdetr-x':
                # Train the model
                model = RTDETR(model_path)
                train_results = model.train(
                    data=data_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    multi_scale = True,
                    optimizer="auto",
                    amp=True,
                    batch = 6,
                    cos_lr=True,
                    #cache=True,
                    lrf=0.001,
                    dropout=0.1,
                    
                    device=device,
                    project=model_output_dir,  # Save training logs and results here
                    name="train_ultralytics",  # Sub-folder name within the project
                    plots=True
                )
            else:
                # Train the model
                model = YOLO(model_path)
                train_results = model.train(
                    data=data_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    multi_scale = True,
                    optimizer="auto",
                    amp=True,
                    batch = 5, # Before 6
                    cos_lr=True,
                    #cache=True,
                    lrf=0.001,
                    dropout=0.1,
                    
                    
                    device=device,
                    project=model_output_dir,  # Save training logs and results here
                    name="train_ultralytics",  # Sub-folder name within the project
                    plots=True
                )

            # Evaluate the model on the validation set and save metrics
            # 1) Evaluate
            metrics = model.val(
                save_json=True,
                plots=True,
                split="test",
                name="test_evaluations",
                conf=0.001,
            )

            # 2) Unpack what mean_results() actually returns:
            #    precision, recall, mAP@0.5, mAP@0.5–0.95
            prec, rec, map50, map50_95 = metrics.mean_results()

            # 3) Grab F1 separately:
            f1_arr = metrics.box.f1
            f1 = float(f1_arr.mean().item())

            # 4) Inference time in ms:
            speed = metrics.speed
            avg_time_ms = float(speed.get('inference', 0.0))

            metrics_data = {
                "Precision":      prec,
                "Recall":         rec,
                "mAP@0.5":        map50,
                "mAP@[0.5:0.95]": map50_95,
                "F1":             f1,
                "Avg inference time (ms)": avg_time_ms,
            }

            # Log the details of the training and evaluation
            log_training_details(model_output_dir, model_name, config, data_path, epochs, imgsz, metrics_data)

            # Update performance metrics for graph plotting
            performance_metrics["model_name"].append(model_name)
            performance_metrics["mAP@0.5"].append(metrics_data["mAP@0.5"])
            performance_metrics["mAP@[0.5:0.95]"].append(metrics_data["mAP@[0.5:0.95]"])
            performance_metrics["Precision"].append(metrics_data["Precision"])
            performance_metrics["Recall"].append(metrics_data["Recall"])
            performance_metrics["F1"].append(metrics_data["F1"])
            performance_metrics["Time"].append(metrics_data["Avg inference time (ms)"])
            

            # Perform object detection on a sample image and save results
            results = model(image_path)
            detection_image_path = os.path.join(model_output_dir, "detection_result.jpg")
            results[0].save(detection_image_path)

            # Export the trained model to ONNX format and save the path
            export_path = model.export(format=format, 
                                       project=model_output_dir,
                                       imgsz=imgsz,
                                       half=True, 
                                       name="model", 
                                       dynamic=dynamic)
            
            print(f"Completed training and evaluation for {model_name}. Results saved in {model_output_dir}")

        except Exception as e:
            print(f"An error occurred while training {model_name}: {e}")

    # Plot performance metrics after training all models
    plot_performance_graph_YOLO(performance_metrics, timestamped_dir)
    print("Training automation completed for all models.")



def YOLO_trainer():
     # Define configurations for different model sizes and other settings
    yolo_model_configs = [
    # YOLOv11 models
    {"name": "yolo11n", "model_path": "yolo11n.pt"},
    {"name": "yolo11s", "model_path": "yolo11s.pt"},
    {"name": "yolo11m", "model_path": "yolo11m.pt"},
    {"name": "yolo11l", "model_path": "yolo11l.pt"},
    {"name": "yolo11x", "model_path": "yolo11x.pt"}, 
    
    # YOLOv8 models
    {"name": "yolo8n", "model_path": "yolov8n.pt"},
    {"name": "yolo8s", "model_path": "yolov8s.pt"},
    {"name": "yolo8m", "model_path": "yolov8m.pt"},
    {"name": "yolo8l", "model_path": "yolov8l.pt"},
    {"name": "yolo8x", "model_path": "yolo8x.pt"}, 
    
    # YOLOv9 models
    {"name": "yolo9t", "model_path": "yolov9t.pt"},
    {"name": "yolo9s", "model_path": "yolov9s.pt"},
    {"name": "yolo9m", "model_path": "yolov9m.pt"},
    {"name": "yolo9c", "model_path": "yolov9c.pt"}, 
    {"name": "yolo9e", "model_path": "yolov9e.pt"}, 

    # YOLOv10 models
    {"name": "yolo10n", "model_path": "yolov10n.pt"},
    {"name": "yolo10s", "model_path": "yolov10s.pt"},
    {"name": "yolo10b", "model_path": "yolov10b.pt"},
    {"name": "yolo10m", "model_path": "yolov10m.pt"},
    {"name": "yolo10l", "model_path": "yolov10l.pt"},


    # RTDETR Models
    {"name": "rtdetr-l", "model_path": "rtdetr-l.pt"},
    {"name": "rtdetr-x", "model_path": "rtdetr-x.pt"},

    # YOLOv12 models (if planning ahead for future versions)
    {"name": "yolo12n", "model_path": "yolo12n.pt"},
    {"name": "yolo12s", "model_path": "yolo12s.pt"},
    {"name": "yolo12m", "model_path": "yolo12m.pt"},
    {"name": "yolo12l", "model_path": "yolo12l.pt"},
    {"name": "yolo12x", "model_path": "yolo12x.pt"}
]
    # General parameters
    data_path = os.path.abspath(os.path.join('..', 'MooTrack360_code', 'datasets', 'training_ready','augmented', 'version_2025-05-01_cropped','yolo_dataset','data.yaml'))
    image_path = os.path.join('..', 'MooTrack360_code','datasets', 'training_ready','augmented','version_2025-05-01_cropped','yolo_dataset', 'val', 'images','7_Kamera20223_frame_51532.jpg')
    output_base_dir = os.path.join('..', 'MooTrack360_code', 'training_outputs')

    epochs = 1
    imgsz = 800

    dynamic = False
    format = "engine" #For NVIDIA GPU's TensorRT ('engine') is preferred

    # Training starts here
    Train_YOLO_models(model_configs=yolo_model_configs, 
                     data_path=data_path, 
                     image_path=image_path, 
                     output_base_dir=output_base_dir, 
                     epochs=epochs, 
                     imgsz=imgsz,
                     dynamic=dynamic,
                     format=format)

if __name__ == "__main__":
    YOLO_trainer()