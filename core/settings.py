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

# Class IDs
# If using fine-tuned 'chicken_model', there is only 1 class (index 0: 'Chicken')
# If using 'yolov8n.pt' (COCO), 'bird' is index 14.
TARGET_CLASS_IDS = [0] if "chicken_model" in MODEL_PATH else [14]
