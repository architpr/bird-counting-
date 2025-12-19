import os

# Base paths
BASE_DIR = "d:/ML project/Bird Counting"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model settings
CUSTOM_MODEL_PATH = os.path.join(OUTPUT_DIR, "chicken_model", "weights", "best.pt")
MODEL_PATH = CUSTOM_MODEL_PATH if os.path.exists(CUSTOM_MODEL_PATH) else "yolov8n.pt" 
# Using nano model for speed or fine-tuned if available
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
TRACKER_CONFIG = "bytetrack.yaml"

# Class IDs for YOLOv8 COCO pretrained
# 14 is 'bird' in COCO
TARGET_CLASS_IDS = [14] 
