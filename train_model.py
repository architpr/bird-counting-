from ultralytics import YOLO
import os

def train_and_evaluate():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # Reduced epochs for prototype, but enough to learn something. 
    # Image size 600 as per dataset description.
    print("Starting training...")
    results = model.train(
        data="data.yaml", 
        epochs=3, 
        imgsz=640, 
        batch=8, 
        project="output", 
        name="chicken_model",
        exist_ok=True,
        verbose=True
    )
    
    print("Training complete.")

    # Validate the model
    print("Starting validation...")
    metrics = model.val()
    
    # Extract Metrics
    # Note: metrics.box.map50 is mAP@50
    # precision and recall are arrays (one per class), since we have 1 class, we take [0] or mean
    
    print("\n" + "="*40)
    print("       MODEL EVALUATION METRICS       ")
    print("="*40)
    
    # Accessing metrics might need internal access or iterating curves for exact F1 at best conf
    # Ultralytics val() returns top-level metrics
    
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    
    # P/R/F1 are curve objects in recent ultralytics versions or attributes
    # We can approximate or print what is available
    
    print(f"mAP@50:    {map50:.4f}")
    print(f"mAP@50-95: {map50_95:.4f}")
    
    # Manually try to get P, R, F1 at best confidence
    # Usually accessible via metrics.mean_results() which returns [P, R, mAP50, mAP50-95]
    p, r, m50, m5095 = metrics.mean_results()
    
    # Calculate F1: 2 * (P * R) / (P + R)
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*40 + "\n")
    
    # Export the model
    path = model.export(format="onnx")
    print(f"Model exported to {path}")
    
    # Update settings to use new model?
    # We can manually ask user or update settings.py
    # For now just save it.

if __name__ == '__main__':
    train_and_evaluate()
