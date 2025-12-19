
import tensorflow as tf
import os
import io
from PIL import Image

def parse_tfrecord_fn(example):
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)

def convert_dataset(tfrecord_path, output_images_dir, output_labels_dir):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    
    count = 0
    for features in parsed_dataset:
        # Save Image
        image_raw = features['image/encoded'].numpy()
        image_name = features['image/filename'].numpy().decode('utf-8')
        
        # Some TFRecords might not have extension in filename
        if not image_name.endswith('.jpg') and not image_name.endswith('.png'):
            image_name += ".jpg"
            
        image_path = os.path.join(output_images_dir, image_name)
        with open(image_path, 'wb') as f:
            f.write(image_raw)
            
        # Process Labels
        # YOLO format: class x_center y_center width height (normalized)
        
        # Check if any objects exist
        if features['image/object/bbox/xmin'].values.shape[0] > 0:
            xmins = features['image/object/bbox/xmin'].values.numpy()
            xmaxs = features['image/object/bbox/xmax'].values.numpy()
            ymins = features['image/object/bbox/ymin'].values.numpy()
            ymaxs = features['image/object/bbox/ymax'].values.numpy()
            labels = features['image/object/class/label'].values.numpy()
            
            label_file = os.path.splitext(image_name)[0] + ".txt"
            label_path = os.path.join(output_labels_dir, label_file)
            
            with open(label_path, 'w') as f:
                for i in range(len(labels)):
                    # Roboflow TFRecord: xmin, ymin, xmax, ymax are NORMALIZED [0,1]
                    xmin = xmins[i]
                    ymin = ymins[i]
                    xmax = xmaxs[i]
                    ymax = ymaxs[i]
                    label = labels[i] - 1  # 1-indexed to 0-indexed for YOLO
                    # CAUTION: 'Chicken' id=1 in pbtxt. YOLO needs 0.
                    
                    # Convert to center_x, center_y, width, height
                    width = xmax - xmin
                    height = ymax - ymin
                    x_center = xmin + width / 2
                    y_center = ymin + height / 2
                    
                    # Clip to [0, 1] just in case
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        count += 1
        if count % 10 == 0:
            print(f"Processed {count} images...", end='\r')
            
    print(f"\nFinished converting {tfrecord_path}. Total: {count}")

if __name__ == "__main__":
    BASE_DIR = "d:/ML project/Bird Counting"
    
    # Train
    TRAIN_RECORD = os.path.join(BASE_DIR, "train", "Chickens.tfrecord")
    TRAIN_IMG_DIR = os.path.join(BASE_DIR, "yolo_data", "images", "train")
    TRAIN_LBL_DIR = os.path.join(BASE_DIR, "yolo_data", "labels", "train")
    
    if os.path.exists(TRAIN_RECORD):
        print(f"Converting Train: {TRAIN_RECORD}")
        convert_dataset(TRAIN_RECORD, TRAIN_IMG_DIR, TRAIN_LBL_DIR)
        
    # Valid
    VAL_RECORD = os.path.join(BASE_DIR, "valid", "Chickens.tfrecord")
    VAL_IMG_DIR = os.path.join(BASE_DIR, "yolo_data", "images", "val")
    VAL_LBL_DIR = os.path.join(BASE_DIR, "yolo_data", "labels", "val")
    
    if os.path.exists(VAL_RECORD):
        print(f"Converting Val: {VAL_RECORD}")
        convert_dataset(VAL_RECORD, VAL_IMG_DIR, VAL_LBL_DIR)
    
    # Test (Optional, usually used for final eval)
    TEST_RECORD = os.path.join(BASE_DIR, "test", "Chickens.tfrecord")
    TEST_IMG_DIR = os.path.join(BASE_DIR, "yolo_data", "images", "test")
    TEST_LBL_DIR = os.path.join(BASE_DIR, "yolo_data", "labels", "test")
    
    if os.path.exists(TEST_RECORD):
        print(f"Converting Test: {TEST_RECORD}")
        convert_dataset(TEST_RECORD, TEST_IMG_DIR, TEST_LBL_DIR)
