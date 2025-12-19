import cv2
import os
import glob

def create_video_from_images(image_folder, output_video_path, fps=5):
    images = sorted(glob.glob(os.path.join(image_folder, "*.jpg"))) + sorted(glob.glob(os.path.join(image_folder, "*.png")))
    
    if not images:
        print(f"No images found in {image_folder}")
        return
    
    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Creating video from {len(images)} images...")
    
    for image in images:
        video.write(cv2.imread(image))
        
    cv2.destroyAllWindows()
    video.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    # Adjust path based on project structure
    # Standard YOLO: train/images
    PROJECT_ROOT = "d:/ML project/Bird Counting"
    IMAGE_DIR = os.path.join(PROJECT_ROOT, "train", "images")
    OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "data", "test_video.mp4")
    
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    
    create_video_from_images(IMAGE_DIR, OUTPUT_VIDEO)
