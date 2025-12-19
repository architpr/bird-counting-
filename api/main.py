from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import shutil
import os
import sys

# Add project root to sys path to import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.pipeline import VideoProcessor
from core.settings import OUTPUT_DIR, DATA_DIR

app = FastAPI(title="Bird Counting API")

# Mount output directory for static file access (video playback)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Service is running"}

@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        input_path = os.path.join(DATA_DIR, f"temp_input_{file.filename}")
        os.makedirs(DATA_DIR, exist_ok=True)
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Define output path
        output_filename = f"processed_{file.filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Process video
        processor = VideoProcessor(input_path, output_path)
        df = processor.process_video()
        
        # Prepare response
        stats = df.to_dict(orient="records")
        video_url = f"/output/{output_filename}"
        
        # Clean up input file (optional, keeping for debug)
        # os.remove(input_path) 
        
        return JSONResponse(content={
            "message": "Processing complete",
            "video_url": video_url,
            "video_path": output_path, # For local access flexibility
            "stats": stats
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
