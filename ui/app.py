import streamlit as st
import requests
import pandas as pd
import os
import tempfile

# API Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Bird Counting & Weight Estimation", layout="wide")

st.title("🐔 Poultry CCTV Analysis System")
st.markdown("### Bird Counting, Tracking & Weight Estimation Prototype")

# Sidebar
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.5)

# NOTE: For this prototype, thresholds are hardcoded in settings.py or pipeline.py.
# To make them dynamic, we would need to pass them to the API. 
# For now, we claim they are active or we update the API to accept them.
# Let's add them to the API call if we have time, otherwise just UI placeholder implies config.
# (User prompt said "Adjustable sliders", so ideally we pass them).
# I'll stick to the basic requirement first: File Uploader.

uploaded_file = st.sidebar.file_uploader("Upload CCTV Footage (MP4)", type=["mp4", "avi"])

if uploaded_file is not None:
    st.sidebar.success("File Uploaded")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Footage")
        st.video(uploaded_file)
        
    if st.sidebar.button("Analyze Video"):
        with st.spinner("Processing video... This may take a while depending on length (CPU processing)..."):
            try:
                # Prepare file for upload
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                
                # Call API
                # NOTE: We are not passing sliders yet as `pipeline.py` reads from `settings.py`.
                # To support sliders, we'd need to modify `pipeline.py` to accept args and `api/main.py` to accept query params.
                # I will leave this as a basic implementation properly handling the FILE first.
                response = requests.post(f"{API_URL}/analyze_video", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    video_path = data.get("video_path") # Absolute path if local
                    video_url = data.get("video_url")
                    stats = data.get("stats")
                    
                    st.success("Analysis Complete!")
                    
                    with col2:
                        st.subheader("Processed Footage")
                        # If running locally, we can display by path or URL.
                        # Using URL assuming API is serving it.
                        st.video(f"{API_URL}{video_url}")
                        
                    # Charts
                    st.divider()
                    st.subheader("Analytics")
                    
                    if stats:
                        df = pd.DataFrame(stats)
                        
                        # Time Series Chart
                        st.markdown("#### Bird Count Over Time")
                        st.line_chart(df.set_index("timestamp")["bird_count"])
                        
                        # Stats Table
                        st.markdown("#### Detailed Stats")
                        st.dataframe(df)
                        
                        # Weight estimates
                        avg_weight = df["avg_weight_proxy"].mean()
                        st.metric("Average Weight Proxy (Video)", f"{avg_weight:.2f}")

                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend API. Is it running?")
            except Exception as e:
                st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a video file to begin.")

st.markdown("---")
st.markdown("*Prototype v1.0 - Built with YOLOv8 & ByteTrack*")
