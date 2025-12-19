import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from .settings import MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, TARGET_CLASS_IDS, TRACKER_CONFIG

class VideoProcessor:
    def __init__(self, source_video_path, output_video_path):
        self.source_video_path = source_video_path
        self.output_video_path = output_video_path
        self.model = YOLO(MODEL_PATH) 
        self.track_history = defaultdict(lambda: [])
        self.unique_ids = set()
        self.frame_data = [] # To store count and weights per frame

    def process_video(self):
        cap = cv2.VideoCapture(self.source_video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {self.source_video_path}")

        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video writer
        out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_idx = 0
        
        # We need to ensure we catch birds. 
        # Since we might not have a trained model for 'chicken', we rely on 'bird' class (14) or just all detections if likely only chickens.
        # But let's assume class 14 for now. PROTOTYPE HACK: If detection is poor, we might need to allow all classes.
        classes_to_track = TARGET_CLASS_IDS
        
        # Iterate
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_idx += 1
            
            # Run YOLOv8 tracking
            # defined via settings
            results = self.model.track(
                frame, 
                persist=True, 
                conf=CONFIDENCE_THRESHOLD, 
                iou=IOU_THRESHOLD,
                tracker=TRACKER_CONFIG,
                classes=classes_to_track,
                verbose=False
            )
            
            current_frame_weights = []
            current_frame_count = 0
            
            # Process results
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                # Visualize
                annotated_frame = results[0].plot()
                
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    
                    # Weight Proxy
                    weight_proxy = float(w * h)
                    current_frame_weights.append(weight_proxy)
                    
                    # Track logic
                    self.unique_ids.add(track_id)
                    
                    # Custom Annotation (Overlay Weight on BBox)
                    # YOLO plot() handles basic ID, but we want to add Weight text
                    # We can draw over the annotated frame
                    # Convert xywh center to top-left for putting text
                    x1 = int(x - w/2)
                    y1 = int(y - h/2)
                    cv2.putText(
                        annotated_frame, 
                        f"W:{weight_proxy:.0f}", 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        2
                    )

            else:
                annotated_frame = frame
                
            current_frame_count = len(current_frame_weights)
            avg_weight = np.mean(current_frame_weights) if current_frame_weights else 0
            
            # Store data
            self.frame_data.append({
                "frame": frame_idx,
                "timestamp": frame_idx / fps,
                "bird_count": current_frame_count,
                "avg_weight_proxy": avg_weight,
                "total_unique_ids": len(self.unique_ids)
            })
            
            # Overlay Global Stats
            cv2.putText(annotated_frame, f"Count: {current_frame_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Total IDs: {len(self.unique_ids)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(annotated_frame)

        cap.release()
        out.release()
        
        return pd.DataFrame(self.frame_data)
