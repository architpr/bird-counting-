from core.pipeline import VideoProcessor
import os

BASE_DIR = "d:/ML project/Bird Counting"
INPUT_VIDEO = os.path.join(BASE_DIR, "data", "sample_video.mp4")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "output", "processed_sample_video.mp4")

print(f"Processing {INPUT_VIDEO}...")
processor = VideoProcessor(INPUT_VIDEO, OUTPUT_VIDEO)
df = processor.process_video()
print(f"Done! Output saved to {OUTPUT_VIDEO}")
print(f"Stats shape: {df.shape}")
