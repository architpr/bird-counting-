@echo off
echo Starting Bird Counting System...
start "BirdCounting API" cmd /k "venv\Scripts\activate && uvicorn api.main:app --reload --port 8000"
timeout /t 5
start "BirdCounting UI" cmd /k "venv\Scripts\activate && streamlit run ui/app.py"
echo System started. API at http://localhost:8000, UI at http://localhost:8501
pause
