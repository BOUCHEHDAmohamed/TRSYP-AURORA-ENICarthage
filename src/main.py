pip install --upgrade ultralytics
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2  # For video frame extraction if needed, install via pip install opencv-python
import pandas as pd
from PIL import Image 
model = YOLO('yolov8m.pt') 
import yaml

# Configuration - Customize these variables for your dataset
dataset_root = '/kaggle/input/traffic-road-object-detection-polish-12k/road_detection/road_detection'  # Root directory of your dataset
train_img_dir = '/kaggle/input/traffic-road-object-detection-polish-12k/road_detection/road_detection/train'         # Relative path to train images
val_img_dir = '/kaggle/input/traffic-road-object-detection-polish-12k/road_detection/road_detection/valid'             # Relative path to validation images
test_img_dir = '/kaggle/input/traffic-road-object-detection-polish-12k/road_detection/road_detection/test'           # Relative path to test images (optional)
class_names = [
    'Car',                          # 0: Vehicles without a trailer
    'Different-Traffic-Sign',       # 1: Other traffic signs (information, order signs)
    'Green-Traffic-Light',          # 2: Green traffic lights for cars only
    'Motorcycle',                   # 3: Motorcycles
    'Pedestrian',                   # 4: People and cyclists
    'Pedestrian-Crossing',          # 5: Pedestrian crossings
    'Prohibition-Sign',             # 6: All prohibition signs
    'Red-Traffic-Light',            # 7: Red traffic lights for cars only
    'Speed-Limit-Sign',             # 8: Speed limit signs
    'Truck',                        # 9: Vehicles with a trailer
    'Warning-Sign'                  # 10: Warning signs
]

nc = len(class_names)                  # Number of classes (auto-calculated)

# Optional: Download script (for Ultralytics datasets, e.g., COCO8)
download_script = 'https://ultralytics.com/assets/coco8.zip'  # Example for COCO8; set to None if not needed

# Build the YAML data structure
data_yaml = {
    'path': dataset_root,                    # Dataset root dir
    'train': train_img_dir,                  # Train images (relative to 'path')
    'val': val_img_dir,                      # Val images (relative to 'path')
    'test': test_img_dir if test_img_dir else None,  # Test images (optional, relative to 'path')
    'names': {i: name for i, name in enumerate(class_names)},  # Class names (index: name)
    # 'nc': nc,                            # Number of classes (optional, auto-detected from 'names')
    'download': download_script              # Optional download script URL
}

# Write to data.yaml file
with open('data.yaml', 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print("data.yaml generated successfully!")
print("Preview:")
print(yaml.dump(data_yaml, default_flow_style=False, sort_keys=False))
train_results = model.train(
    data='/kaggle/working/data.yaml',  # Path to YAML file defining dataset paths and classes
    epochs=100,                # Number of epochs; monitor for early stopping if overfitting
    imgsz=640,                 # Input image size (square, resizes images)
    batch=16,                  # Batch size; reduce if GPU memory is low (e.g., to 8)
    workers=4,                 # DataLoader workers; adjust based on CPU cores
    device=0,                  # GPU device (0 for single GPU); use 'cpu' if no GPU
    name='traffic_yolov8m',  # Output directory name
    patience=20,               # Early stopping patience (epochs without improvement)
    optimizer='AdamW',         # Optimizer; alternatives: 'SGD', 'Adam'
    lr0=0.01,                  # Initial learning rate
    momentum=0.937,            # SGD momentum
    weight_decay=0.0005,       # Weight decay
    amp=True,                  # Automatic Mixed Precision for faster training
    plots=True                 # Generate plots during training (losses, metrics)
)  
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
model = YOLO('/kaggle/working/runs/detect/traffic_yolov8m/weights/best.pt')

# Define the path to the test images folder from your dataset
test_images_path = '/kaggle/input/traffic-road-object-detection-polish-12k/road_detection/road_detection/test'

# Counter to limit processing to 5 images
image_count = 0
max_images = 5

# Iterate over images in the test folder
for image_name in os.listdir(test_images_path):
    if image_count >= max_images:  # Stop after 5 images
        break
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Support common image formats
        image_path = os.path.join(test_images_path, image_name)
        
        # Run inference on the image
        results = model(image_path, save=True, imgsz=640)
                # Get the path where the annotated image is saved
        output_path = os.path.join(results[0].save_dir, image_name)
        print(f"Processed {image_name}. Annotated image saved at: {output_path}")
        
        # Display the annotated image
        img = Image.open(output_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')  # Hide axes
        plt.title(f"Annotated Image: {image_name}")
        plt.show()
        
        image_count += 1

# Optionally, evaluate the model on the entire test set for metrics
val_results = model.val(
    data='/kaggle/working/data.yaml',
    split='test',
    imgsz=640
)

# Print evaluation metrics
print(f"mAP@0.5: {val_results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {val_results.box.map:.4f}")





