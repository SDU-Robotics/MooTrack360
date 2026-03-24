import os
import random
import shutil
import json
from collections import defaultdict
import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np
np.bool = bool
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from dotenv import load_dotenv
from tqdm import tqdm
import glob

"""
This script converts the original downloaded annotations from Azure and make two coresponding dataset: One for YOLO- and one for COCO formatted models.

The path to the original dataset should be passed (data_dir), change this according to your needs.

The path to the output folder should not be modified. 

The following parameters can be changes:
- train_ratio = 0.7
- val_ratio = 0.2
- test_ratio = 0.1
- seed = 42

Augmentations can be applied by setting the augment = True and then setting num_of_augmentations = integer.
For each image the algorithm will randomly create x number of "realistic" augmentations based on what num_of_augmentations is set to.

If the augment = True the generated dataset will be in the folder ".../training_ready/augmmented/{Name_of_dataset}" if it is set to false,
the generated dataset will be in the folder ".../training_ready/non_augmmented/{Name_of_dataset}"

"""

TEST = False  # used for debugging augmentations

def make_yolo_coco_datasets(data_dir:str, 
                            output_base_dir:str, 
                            train_ratio:float = 0.7, 
                            val_ratio:float = 0.2, 
                            test_ratio:float = 0.1, 
                            seed:int = 42, 
                            augment:bool = False, 
                            num_of_augmentations:int = 0):

    # Validate that the ratios sum up to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    data_dir_name = os.path.basename(data_dir)

    # Construct output dirs depending on augmentation flag
    if augment:
        output_yolo_dir = os.path.join(output_base_dir, 'augmented', data_dir_name, 'yolo_dataset')
        output_coco_dir = os.path.join(output_base_dir, 'augmented', data_dir_name, 'coco_dataset')
    else:
        output_yolo_dir = os.path.join(output_base_dir, 'non_augmented', data_dir_name, 'yolo_dataset')
        output_coco_dir = os.path.join(output_base_dir, 'non_augmented', data_dir_name, 'coco_dataset')

    # Check if dataset already exists and exit if it does
    if os.path.exists(output_yolo_dir) or os.path.exists(output_coco_dir):
        print(f"Dataset for '{data_dir_name}' already exists. Exiting.")
        return

    # Make sure directories exist
    os.makedirs(output_yolo_dir, exist_ok=True)
    os.makedirs(output_coco_dir, exist_ok=True)

    # Set random seed
    random.seed(seed)

    # Automatically find the normalized_*.json file in data_dir
    annotation_files = glob.glob(os.path.join(data_dir, "normalized_*.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No 'normalized_*.json' annotation file found in {data_dir}")
    annotation_file_path = annotation_files[0]  # Use the first match

    # Load COCO annotation file
    with open(annotation_file_path, 'r') as f:
        coco_data = json.load(f)

    # Build images_info dictionary
    images_info = {}
    for img in coco_data['images']:
        images_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    # Build annotations dictionary
    id_to_label = {cat['id']: cat['name'] for cat in coco_data['categories']}
    annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        cat_id = ann['category_id']
        label = id_to_label[cat_id]
        # If not, you need to adjust this part accordingly.
        x_center, y_center, w, h = ann['bbox']
        annotations[image_id].append({
            'label': label,
            'bbox': [x_center, y_center, w, h]
        })

    # Get list of image IDs
    image_ids = list(images_info.keys())
    total_images = len(image_ids)

    # Shuffle and split image IDs
    random.shuffle(image_ids)

    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    test_end = val_end + int(total_images * test_ratio)

    # Adjust for rounding
    if test_end < total_images:
        test_end = total_images

    train_ids = image_ids[:train_end]
    val_ids = image_ids[train_end:val_end]
    test_ids = image_ids[val_end:test_end]

    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    # Extract unique labels
    labels_set = set()
    for img_id in annotations:
        for ann in annotations[img_id]:
            labels_set.add(ann['label'])
    labels = sorted(list(labels_set))
    label_to_id = {label: idx + 1 for idx, label in enumerate(labels)} 
    
    def create_yolo_dataset(split_name, image_ids, images_info, annotations, output_dir):
        images_output_dir = os.path.join(output_dir, split_name, 'images')
        labels_output_dir = os.path.join(output_dir, split_name, 'labels')
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(labels_output_dir, exist_ok=True)

        # Define augmentation sequences
        seq1 = iaa.Sequential([
            iaa.Sometimes(0.8,iaa.OneOf([iaa.CropToFixedSize(width=2000, height=2000, position=(iap.Uniform(0.0, 0.3), iap.Uniform(0.0, 0.3))),
                                        iaa.CropToFixedSize(width=1280, height=1280, position=(iap.Uniform(0.0, 0.4), iap.Uniform(0.0, 0.1))),
                                        iaa.CropToFixedSize(width=1280, height=1280, position=(iap.Uniform(0.0, 0.4), iap.Uniform(0.3, 0.4))),
                                        iaa.CropToFixedSize(width=800, height=800, position=(iap.Uniform(0.0, 0.65), iap.Uniform(0.0, 0.25))),
                                        iaa.CropToFixedSize(width=800, height=800, position=(iap.Uniform(0.0, 0.65), iap.Uniform(0.4, 0.65))),
                                        iaa.CropToFixedSize(width=640, height=640, position=(iap.Uniform(0.0, 0.75), iap.Uniform(0.0, 0.35))),
                                        iaa.CropToFixedSize(width=640, height=640, position=(iap.Uniform(0.0, 0.75), iap.Uniform(0.45, 0.75)))]),
                        ),
            iaa.Sometimes(0.3, iaa.CropAndPad(percent=(0.0, -0.4), pad_mode="linear_ramp", pad_cval=(0, 255))),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Rot90(k=(0, 4), keep_size=False)
        ], random_order=False)

        seq2 = iaa.SomeOf((3, 6), [
            iaa.CoarseDropout((0.01, 0.02), size_percent=(0.125, 0.175)),
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.PerspectiveTransform(scale=(0.01, 0.1)),
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.LinearContrast((0.75, 1.1)),
            iaa.Grayscale(1.0),
            iaa.MultiplySaturation((0.8, 1.25)),
            iaa.pillike.EnhanceSharpness(),
            iaa.ElasticTransformation(alpha=(0, 30), sigma=(5, 10))
        ])

        for image_id in tqdm(image_ids, desc=f"Processing YOLO {split_name}"):
            image_info = images_info[image_id]
            image_name = image_info['file_name']

            # Copy image to output directory
            src_image_path = os.path.join(data_dir, str(image_id) +"_"+ image_name)
            dst_image_path = os.path.join(images_output_dir, str(image_id) +"_"+ image_name)
            if not os.path.exists(src_image_path):
                print(f"Warning: Image file {image_name} does not exist.")
                continue
            shutil.copyfile(src_image_path, dst_image_path)

            # Create label file
            label_file_name = os.path.splitext(str(image_id) +"_"+image_name)[0] + '.txt'
            label_file_path = os.path.join(labels_output_dir, label_file_name)
            with open(label_file_path, 'w') as f:
                if image_id in annotations:
                    # if name has _augmented....:
                    for ann in annotations[image_id]:
                        label = ann['label']
                        # ann['bbox'] = [x, y, w, h], top-left normalized
                        x, y, w_box, h_box = ann['bbox']
                        x_center = x + w_box / 2.0
                        y_center = y + h_box / 2.0
                        label_id = label_to_id[label] - 1  # YOLO labels start from 0

                        f.write(f"{label_id} {x_center} {y_center} {w_box} {h_box}\n")


            # Perform data augmentation if augment flag is set
            if augment and split_name in ['train', 'val']: 
                image = cv2.imread(src_image_path)
                if image_id in annotations:
                    bbs = []
                    for ann in annotations[image_id]:
                        x_tl, y_tl, w_n, h_n = ann['bbox']
                        height = image_info['height']
                        width = image_info['width']

                        w_abs = w_n * width
                        h_abs = h_n * height
                        x1 = x_tl * width
                        y1 = y_tl * height
                        x2 = x1 + w_abs
                        y2 = y1 + h_abs

                        bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label_to_id[ann['label']] - 1))
                    bbs_on_image = BoundingBoxesOnImage(bbs, shape=image.shape)

                    for i in range(num_of_augmentations):
                        if i < num_of_augmentations / 2:
                            augmented_image, augmented_annotations = seq1(image=image, bounding_boxes=bbs_on_image)
                        else:
                            augmented_image_temp, augmented_annotations_temp = seq1(image=image, bounding_boxes=bbs_on_image)
                            augmented_image, augmented_annotations = seq2(image=augmented_image_temp, bounding_boxes=augmented_annotations_temp)

                        augmented_image_height, augmented_image_width, _ = augmented_image.shape
                        augmented_annotations = augmented_annotations.remove_out_of_image().clip_out_of_image()

                        if any((x.is_fully_within_image(augmented_image.shape) or x.is_partly_within_image(augmented_image.shape)) for x in augmented_annotations):
                            if TEST:
                                for x in augmented_annotations:
                                    augmented_image = x.draw_on_image(augmented_image, size=2, color=[0, 0, 255])
                                plt.figure(figsize=(8, 6))
                                plt.imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
                                plt.axis('off')
                                plt.title("After Augmentation")
                                plt.show()
                                break

                            # Save augmented image
                            augmented_image_name = f"{os.path.splitext(image_name)[0]}_augmented_{i}.jpg"
                            augmented_image_path = os.path.join(images_output_dir, augmented_image_name)
                            cv2.imwrite(augmented_image_path, augmented_image)

                            # Convert augmented bbs back to normalized YOLO format and save
                            augmented_label_file_path = os.path.join(labels_output_dir, f"{os.path.splitext(image_name)[0]}_augmented_{i}.txt")
                            with open(augmented_label_file_path, 'w') as label_file:
                                for annotation in augmented_annotations.bounding_boxes:
                                    # Convert absolute coords back to normalized YOLO format
                                    box_w = annotation.x2 - annotation.x1
                                    box_h = annotation.y2 - annotation.y1
                                    x_c = annotation.x1 + box_w / 2.0
                                    y_c = annotation.y1 + box_h / 2.0
                                    
                                    x_center_norm = x_c / augmented_image_width
                                    y_center_norm = y_c / augmented_image_height
                                    width_norm = box_w / augmented_image_width
                                    height_norm = box_h / augmented_image_height
                                    
                                    boundingBoxArea = 1.0 * width_norm * height_norm
                                    if x_center_norm > 0.98 or x_center_norm < 0.02 or y_center_norm > 0.98 or y_center_norm < 0.03:
                                        continue
                                    elif boundingBoxArea < 0.001 or width_norm < 0.005 or height_norm < 0.005:
                                        continue
                                    elif width_norm > 0.95 and height_norm > 0.95:
                                        continue
                                    else:
                                        label_file.write(f"{annotation.label} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

    def create_coco_dataset(split_name, image_ids, images_info, annotations, output_dir):
        images_list = []
        annotations_list = []
        annotation_id = 1  # COCO annotation IDs start at 1
        image_id_counter = max(images_info.keys()) + 1

        os.makedirs(output_dir, exist_ok=True)
        images_output_dir = os.path.join(output_dir, split_name)
        os.makedirs(images_output_dir, exist_ok=True)

        # Define augmentation sequences
        seq1 = iaa.Sequential([
            iaa.Sometimes(0.8,iaa.OneOf([iaa.CropToFixedSize(width=1280, height=1280, position=(iap.Uniform(0.0, 0.4), iap.Uniform(0.0, 0.1))),
                                        iaa.CropToFixedSize(width=1280, height=1280, position=(iap.Uniform(0.0, 0.4), iap.Uniform(0.3, 0.4))),
                                        iaa.CropToFixedSize(width=800, height=800, position=(iap.Uniform(0.0, 0.65), iap.Uniform(0.0, 0.25))),
                                        iaa.CropToFixedSize(width=800, height=800, position=(iap.Uniform(0.0, 0.65), iap.Uniform(0.4, 0.65)))]),
                        ),
            iaa.Sometimes(0.3, iaa.CropAndPad(percent=(0.0, -0.4), pad_mode="linear_ramp", pad_cval=(0, 255))),
            #iaa.Resize({"height": "keep-aspect-ratio", "width": 800}),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Rot90(k=(0, 4), keep_size=False)
        ], random_order=False)

        seq2 = iaa.SomeOf((3, 6), [
            iaa.CoarseDropout((0.01, 0.02), size_percent=(0.125, 0.175)),
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.PerspectiveTransform(scale=(0.01, 0.1)),
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.LinearContrast((0.75, 1.1)),
            iaa.Grayscale(1.0),
            iaa.MultiplySaturation((0.8, 1.25)),
            iaa.pillike.EnhanceSharpness(),
            iaa.ElasticTransformation(alpha=(0, 30), sigma=(5, 10))
        ],random_order=True)

        categories_list = [{'id': id, 'name': name} for name, id in label_to_id.items()]

        for image_id in tqdm(image_ids, desc=f"Processing COCO {split_name}"):
            image_info = images_info[image_id]
            image_name = image_info['file_name']
            width = image_info['width']
            height = image_info['height']

            images_list.append({
                'id': image_id,
                'file_name': image_name,
                'width': width,
                'height': height
            })

            if image_id in annotations:
                # Convert top-left normalized bbox to COCO pixel format
                for ann in annotations[image_id]:
                    label = ann['label']
                    x, y, w_box, h_box = ann['bbox']
                    x_min = x * width
                    y_min = y * height
                    w_pixel = w_box * width
                    h_pixel = h_box * height

                    annotations_list.append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': label_to_id[label],
                        'bbox': [x_min, y_min, w_pixel, h_pixel],
                        'area': w_pixel * h_pixel,
                        'iscrowd': 0
                    })
                    annotation_id += 1

            # Perform data augmentation if augment flag is set
            if augment:
                src_image_path = os.path.join(data_dir, image_name)
                image = cv2.imread(src_image_path)
                if image_id in annotations:
                    bbs = []
                    for ann in annotations[image_id]:
                        x_tl, y_tl, w_n, h_n = ann['bbox']
                        height = image_info['height']
                        width = image_info['width']

                        w_abs = w_n * width
                        h_abs = h_n * height
                        x1 = x_tl * width
                        y1 = y_tl * height
                        x2 = x1 + w_abs
                        y2 = y1 + h_abs

                        bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label_to_id[ann['label']]))
                    bbs_on_image = BoundingBoxesOnImage(bbs, shape=image.shape)

                    for i in range(num_of_augmentations):
                        if i < num_of_augmentations / 2:
                            augmented_image, augmented_annotations = seq1(image=image, bounding_boxes=bbs_on_image)
                        else:
                            augmented_image_temp, augmented_annotations_temp = seq1(image=image, bounding_boxes=bbs_on_image)
                            augmented_image, augmented_annotations = seq2(image=augmented_image_temp, bounding_boxes=augmented_annotations_temp)

                        augmented_image_height, augmented_image_width, _ = augmented_image.shape
                        augmented_annotations = augmented_annotations.remove_out_of_image().clip_out_of_image()

                        if any((x.is_fully_within_image(augmented_image.shape) or x.is_partly_within_image(augmented_image.shape)) for x in augmented_annotations):
                            augmented_image_id = image_id_counter
                            image_id_counter += 1
                            aug_image_name = f"{os.path.splitext(image_name)[0]}_augmented_{i}.jpg"
                            images_list.append({
                                'id': augmented_image_id,
                                'file_name': aug_image_name,
                                'width': augmented_image_width,
                                'height': augmented_image_height
                            })

                            for annotation in augmented_annotations.bounding_boxes:
                                x1 = annotation.x1
                                y1 = annotation.y1
                                w_aug = annotation.x2 - annotation.x1
                                h_aug = annotation.y2 - annotation.y1

                                annotations_list.append({
                                    'id': annotation_id,
                                    'image_id': augmented_image_id,
                                    'category_id': annotation.label,
                                    'bbox': [x1, y1, w_aug, h_aug],
                                    'area': w_aug * h_aug,
                                    'iscrowd': 0
                                })
                                annotation_id += 1

                            augmented_image_path = os.path.join(images_output_dir, aug_image_name)
                            cv2.imwrite(augmented_image_path, augmented_image)

        coco_format = {
            'images': images_list,
            'annotations': annotations_list,
            'categories': categories_list
        }

        # Save COCO JSON file
        json_file_path = os.path.join(output_dir, f'instances_{split_name}.json')
        with open(json_file_path, 'w') as f:
            json.dump(coco_format, f)

        # Copy original images to output directory
        for image_id in image_ids:
            image_info = images_info[image_id]
            image_name = image_info['file_name']
            src_image_path = os.path.join(data_dir, image_name)
            dst_image_path = os.path.join(images_output_dir, image_name)
            if not os.path.exists(src_image_path):
                continue
            if not os.path.exists(dst_image_path):
                shutil.copyfile(src_image_path, dst_image_path)

    # Generate YOLO and COCO datasets
    for split_name, ids in splits.items():
        create_yolo_dataset(split_name, ids, images_info, annotations, output_yolo_dir)
        #create_coco_dataset(split_name, ids, images_info, annotations, output_coco_dir)

    def create_data_yaml(output_dir, labels):
        data_yaml_path = os.path.join(output_dir, 'data.yaml')

        data_yaml = {
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(labels),
            'names': labels
        }

        class FlowStyleList(list):
            pass

        def flow_style_list_representer(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

        yaml.add_representer(FlowStyleList, flow_style_list_representer)
        data_yaml['names'] = FlowStyleList(labels)

        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        print(f"data.yaml file has been created at {data_yaml_path}")

    create_data_yaml(output_yolo_dir, labels)
    print("Datasets have been successfully generated in YOLO and COCO formats!")

if __name__ == "__main__":
    load_dotenv()

    # Parameters
    data_dir = "MooTrack360_code/datasets/raw-images/version_2025-05-01_cropped"
    output_base_dir = "MooTrack360_code/datasets/training_ready"

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    seed = 100  

    augment = True
    num_of_augmentations = 10 

    make_yolo_coco_datasets(data_dir=data_dir,
                            output_base_dir=output_base_dir,
                            train_ratio=train_ratio,
                            val_ratio=val_ratio,
                            test_ratio=test_ratio,
                            seed=seed,
                            augment=augment,
                            num_of_augmentations=num_of_augmentations)
