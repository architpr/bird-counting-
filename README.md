# Bird Counting and Weight Estimation System 🐔

A computer vision prototype for counting birds and estimating their weight from CCTV footage using YOLOv8 and ByteTrack.

## Features
- **Detection**: YOLOv8 (Nano) for fast bird detection.
- **Tracking**: ByteTrack to handle occlusion and improved counting.
- **Weight Estimation**: Relative Weight Index ($Width \times Height$) as a proxy.
- **Interactive UI**: Streamlit dashboard for easy video analysis.
- **Rest API**: FastAPI backend for processing.

## Project Structure
```
.
├── api/             # FastAPI Backend
│   └── main.py
├── core/            # computer Vision Logic
│   ├── pipeline.py
│   └── settings.py
├── ui/              # Streamlit Frontend
│   └── app.py
├── data/            # Data storage (input/output)
├── requirements.txt
└── README.md
```

## Setup

1.  **Environment Setup**:
    Ensure you have Python 3.8+ installed.
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You need to run both the Backend (FastAPI) and Frontend (Streamlit).

### 1. Start the Backend API
Run this in one terminal:
```bash
uvicorn api.main:app --reload --port 8000
```
Check health: [http://localhost:8000/health](http://localhost:8000/health)

### 2. Start the Frontend UI
Run this in a separate terminal:
```bash
streamlit run ui/app.py
```
Open your browser at [http://localhost:8501](http://localhost:8501).

### 3. Analyze Video
1.  Open the Streamlit App.
2.  Upload a CCTV MP4/AVI file.
3.  Click "Analyze Video".
4.  View results and charts.

## API Usage (cURL)

You can also use the API directly:

```bash
curl -X POST "http://localhost:8000/analyze_video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/video.mp4"
```

## Assumptions & Limitations
- **Weight Proxy**: The weight is strictly a heuristic ($W \times H$) in pixels. Real-world usage requires calibration with ground truth weights.
- **Model**: Uses pretrained YOLOv8n (COCO 'bird' class). Fine-tuning on the specific "Chickens" dataset is recommended for production accuracy.
- **Performance**: Processing speed depends on hardware (GPU recommended).

## License
MIT
