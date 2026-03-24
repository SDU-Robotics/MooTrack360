import cv2
import json
import os
import re
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

"""
This script displays annotated bounding boxes on images downloaded from Azure and,
once you exit the viewer, computes and plots various statistics:
- Composite Camera Label: last 3 digits of the full camera ID + stable (e.g., "123_stable2")
- Date: extracted from the file name (YYYY-MM-DD)
- Day/Night: based on whether the image is color (Day) or grayscale (Night)
- Total number of cows and breakdown (standing: class 0, lying: class 1)

Navigation:
  Arrow keys (or a/d) to navigate and q or ESC to exit the viewer.
"""

# ------------------------------
# Settings and paths
# ------------------------------
main_dir = Path("MooTrack360_code", "datasets", "raw-images")
dataset_dir = main_dir / "version_2025-02-06_cropped"
json_file = dataset_dir / "normalized_annotations_02_06.json"
images_dir = dataset_dir

# ------------------------------
# Load annotation data
# ------------------------------
with open(json_file, 'r') as f:
    data = json.load(f)

images_info = data["images"]
annotations = data["annotations"]
categories = data["categories"]

image_to_annotations = {}
for ann in annotations:
    image_id = ann["image_id"]
    image_to_annotations.setdefault(image_id, []).append(ann)

# ------------------------------
# Assign colors for categories
# ------------------------------
category_colors = {
    1: (0, 255, 0),   # Green
    2: (255, 0, 0),   # Blue
    3: (0, 0, 255)    # Red
}

cat_id_to_info = {}
for cat in categories:
    cat_id = cat["id"]
    cat_name = cat["name"]
    color = category_colors.get(cat_id, (255,255,255))
    cat_id_to_info[cat_id] = (cat_name, color)

def draw_bboxes(img, anns, img_width, img_height):
    for ann in anns:
        cat_id = ann["category_id"]
        cat_name, color = cat_id_to_info.get(cat_id, ("Unknown", (255,255,255)))
        # Extract normalized bbox and scale to image size
        x, y, w, h = ann["bbox"]
        x1 = int(x * img_width)
        y1 = int(y * img_height)
        w_box = int(w * img_width)
        h_box = int(h * img_height)
        x2 = x1 + w_box
        y2 = y1 + h_box
        # Draw the bounding box and category name
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, cat_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# ------------------------------
# Image viewing loop
# ------------------------------
idx = 0
num_images = len(images_info)

while True:
    img_info = images_info[idx]
    img_path = os.path.join(images_dir, img_info["file_name"])
    if not os.path.exists(img_path):
        print(f"Image {img_path} not found! Skipping...")
        idx = (idx + 1) % num_images
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}.")
        idx = (idx + 1) % num_images
        continue

    img_id = img_info["id"]
    img_height = img_info["height"]
    img_width = img_info["width"]

    anns_for_image = image_to_annotations.get(img_id, [])
    displayed_img = img.copy()
    displayed_img = draw_bboxes(displayed_img, anns_for_image, img_width, img_height)

    # Scale the image down to 50% of its original size
    displayed_img = cv2.resize(displayed_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("Image Viewer", displayed_img)
    key = cv2.waitKey(0)

    if key == 27 or key == ord('q'):  # ESC or q to quit
        break
    elif key == 81 or key == ord('a'):  # Left arrow or 'a'
        idx = (idx - 1) % num_images
    elif key == 83 or key == ord('d'):  # Right arrow or 'd'
        idx = (idx + 1) % num_images

cv2.destroyAllWindows()

# ------------------------------
# Helper functions for statistics
# ------------------------------
def extract_camera_id(file_name):
    """Extracts the full camera id from the file name (expects a part like '0002D1...')."""
    parts = file_name.split('_')
    for part in parts:
        if part.startswith("0002D1"):
            return part
    return "Unknown"

def extract_date(file_name):
    """Extracts a date from the beginning of the file name (YYYY_MM_DD or YYYYMMDD) and returns it as YYYY-MM-DD."""
    match = re.match(r"^(\d{4}[_]?\d{2}[_]?\d{2})", file_name)
    if match:
        date_str = match.group(1)
        if "_" in date_str:
            parts = date_str.split('_')
        else:
            parts = [date_str[:4], date_str[4:6], date_str[6:]]
        return f"{parts[0]}-{parts[1]}-{parts[2]}"
    return "Unknown"

def is_daytime(image, threshold=20):
    """
    Determines whether an image is considered 'daytime' based on brightness.
    The image is first converted to grayscale, and its mean pixel intensity is computed.
    If the mean brightness is above the threshold, it is classified as Day, otherwise Night.
    
    Args:
        image (numpy.ndarray): The BGR image.
        threshold (int, optional): Brightness threshold (0-255). Defaults to 80.
    
    Returns:
        bool: True if image is day, False if night.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    return brightness > threshold


# ------------------------------
# Load stable mapping and camera ordering from YAML
# ------------------------------
yaml_path = os.path.join("VelKoTek", "src", "configs", "stable_configs.yaml")
camera_to_stable = {}
camera_order = {}  # Mapping from full camera id to (stable_number, camera_number)
try:
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    for stable, stable_data in yaml_data.items():
        for area, area_data in stable_data.items():
            for cam_key, cam_info in area_data.items():
                cam_id = cam_info.get("id")
                if cam_id:
                    camera_to_stable[cam_id] = stable
                    try:
                        stable_number = int(re.search(r'\d+', stable).group())
                    except Exception as e:
                        stable_number = 999
                    try:
                        camera_number = int(re.search(r'\d+', cam_key).group())
                    except Exception as e:
                        camera_number = 999
                    camera_order[cam_id] = (stable_number, camera_number)
except Exception as e:
    print("Error loading YAML file for stable mapping:", e)

# ------------------------------
# Compute statistics with sorted composite camera labels
# ------------------------------
camera_counts = {}
date_counts = {}
stable_counts = {}
day_night_counts = {"Day": 0, "Night": 0}
composite_order = {}  # Mapping from composite camera label to ordering tuple

for img_info in images_info:
    file_name = img_info["file_name"]
    full_cam_id = extract_camera_id(file_name)
    if full_cam_id == "Unknown":
        composite_cam_label = "Unknown"
        order_tuple = (999, 999)
        stable = "Unknown"
    else:
        short_cam_id = full_cam_id[-3:]
        stable = camera_to_stable.get(full_cam_id, "Unknown")
        composite_cam_label = f"{short_cam_id}_{stable}"
        order_tuple = camera_order.get(full_cam_id, (999, 999))
    camera_counts[composite_cam_label] = camera_counts.get(composite_cam_label, 0) + 1
    composite_order[composite_cam_label] = order_tuple

    # Count images per date
    date_str = extract_date(file_name)
    date_counts[date_str] = date_counts.get(date_str, 0) + 1

    # Count images per stable
    stable_counts[stable] = stable_counts.get(stable, 0) + 1

    # Determine day vs night by reading the image
    img_path = os.path.join(images_dir, file_name)
    img = cv2.imread(img_path)
    if img is not None:
        if is_daytime(img, threshold=61):
            day_night_counts["Day"] += 1
        else:
            day_night_counts["Night"] += 1

# Sort composite camera labels (first by stable then by camera number)
sorted_camera_labels = sorted(camera_counts.keys(), key=lambda label: composite_order[label])

# Sort the dates chronologically (assuming format "YYYY-MM-DD")
sorted_dates = sorted(date_counts.keys(), key=lambda d: datetime.datetime.strptime(d, "%Y-%m-%d") if d != "Unknown" else datetime.datetime(9999, 12, 31))

# Compute cow counts (assuming class 0: standing, class 1: lying)
total_cows = 0
standing_cows = 0
lying_cows = 0
for ann in annotations:
    cat = ann["category_id"]
    if cat in [2, 3]:
        total_cows += 1
        if cat == 2:
            standing_cows += 1
        elif cat == 3:
            lying_cows += 1

# ------------------------------
# Plot statistics
# ------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot: Images per Composite Camera Label (sorted)
cam_labels = sorted_camera_labels
cam_counts = [camera_counts[label] for label in cam_labels]
axs[0, 0].bar(cam_labels, cam_counts)
axs[0, 0].set_title("Images per Camera (sorted by Stable & Camera Number)")
axs[0, 0].set_xlabel("Camera (Last 3 digits_Stable)")
axs[0, 0].set_ylabel("Number of Images")
axs[0, 0].tick_params(axis='x', rotation=45)

# Plot: Images per Stable
axs[0, 1].bar(list(stable_counts.keys()), list(stable_counts.values()))
axs[0, 1].set_title("Images per Stable")
axs[0, 1].set_xlabel("Stable")
axs[0, 1].set_ylabel("Number of Images")

# Plot: Images per Date (sorted chronologically)
date_counts_sorted = [date_counts[d] for d in sorted_dates]
axs[1, 0].bar(sorted_dates, date_counts_sorted)
axs[1, 0].set_title("Images per Date")
axs[1, 0].set_xlabel("Date")
axs[1, 0].set_ylabel("Number of Images")
axs[1, 0].tick_params(axis='x', rotation=45)

# Plot: Day vs Night (pie chart)
axs[1, 1].pie(list(day_night_counts.values()), labels=list(day_night_counts.keys()), autopct='%1.1f%%', startangle=90)
axs[1, 1].set_title("Day vs Night Images")

plt.tight_layout()
plt.show()

# ------------------------------
# Print cow statistics
# ------------------------------
print("Total number of cows:", total_cows)
print("Number of standing cows (class=0):", standing_cows)
print("Number of lying cows (class=1):", lying_cows)