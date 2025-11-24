import os
import re
import tempfile
from datetime import timedelta

import cv2
import google.generativeai as genai
import numpy as np
import streamlit as st
import time
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

st.set_page_config(
    page_title="Weapon Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .stFileUploader {
        padding: 20px;
        border-radius: 15px;
        border: 1px dashed #e0e0e0;
    }
    
    .stButton>button {
        width: 100%;
        background-color: #007AFF;
        color: white;
        border: none;
        border-radius: 12px;
        height: 45px;
        font-weight: 500;
        font-size: 16px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0051a8;
        transform: scale(1.01);
        color: white;
    }
    
    .status-container {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }
    
    .status-indicator {
        padding: 10px 24px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }

    .report-card {
        background-color: transparent;
        border-left: 6px solid #007AFF;
        padding: 10px 0px 10px 25px;
        margin-top: 20px;
    }

    .report-header {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 15px;
        letter-spacing: 0.5px;
        color: inherit;
    }

    .report-content {
        font-size: 18px !important;
        line-height: 1.8 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: inherit;
    }

    .highlight-text {
        background-color: rgba(255, 235, 59, 0.5);
        color: #000000 !important;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight:500;
        margin: 0 2px;
    }
    </style>
    """, unsafe_allow_html=True)

def render_status(detected):
    if detected:
        return """
        <div class="status-container">
            <div class="status-indicator" style="background-color: #FFF5F5; color: #D32F2F; border: 1px solid #FFEBEE;">
                THREAT FOUND
            </div>
        </div>
        """
    else:
        return """
        <div class="status-container">
            <div class="status-indicator" style="background-color: #F7F9FA; color: #007AFF; border: 1px solid #E1E8ED;">
                NO THREAT
            </div>
        </div>
        """

@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        return None

def process_video_frame(frame, model, process_width=640):
    height, width = frame.shape[:2]
    ratio = process_width / width
    new_height = int(height * ratio)
    resized_frame = cv2.resize(frame, (process_width, new_height))
    results = model(resized_frame, conf=0.4, verbose=False)
    return results

def analyze_detection(results):
    detected = False
    max_confidence = 0.0
    
    if len(results[0].boxes) > 0:
        detected = True
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > max_confidence:
                max_confidence = conf
    
    return detected, max_confidence

def generate_ai_analysis(image_array, api_key):
    try:
        genai.configure(api_key=api_key)
        ai_model = genai.GenerativeModel('gemini-2.5-flash')
        pil_image = Image.fromarray(image_array)
        
        prompt = (
            "Analyze this image of a detected weapon. "
            "Provide a clear, simple summary. "
            "CRITICAL INSTRUCTION: Use **bold markdown** for the most important Sentence!! (like weapon name or threat level or what's happening). "
            "Do not use code blocks. Keep it 3-7 lines."
        )
        
        response = ai_model.generate_content([prompt, pil_image])
        formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<span class="highlight-text">\1</span>', response.text)
        
        return formatted_text
    except Exception as e:
        raise Exception(f"AI Analysis Failed: {e}")

model_path = 'model/best.pt'
model = load_model(model_path)

st.markdown("## Detection Weapon ")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")

if uploaded_file is not None and model is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    stop_button = st.button("End Session")
    
    if not stop_button:
        status_placeholder = st.empty()
        st_frame = st.empty()
        
        PROCESS_WIDTH = 640
        fps_input = cap.get(cv2.CAP_PROP_FPS)
        if fps_input <= 0:
            fps_input = 30
        
        SKIP_FRAMES = max(1, int(fps_input / 3))
        
        frame_count = 0
        highest_confidence = 0.0
        best_frame_rgb = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % SKIP_FRAMES != 0:
                continue
            
            results = process_video_frame(frame, model, PROCESS_WIDTH)
            detected, current_max_conf = analyze_detection(results)
            
            if detected and current_max_conf > highest_confidence:
                highest_confidence = current_max_conf
                annotated_best = results[0].plot()
                best_frame_rgb = cv2.cvtColor(annotated_best, cv2.COLOR_BGR2RGB)
            
            annotated_frame = results[0].plot()
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            status_placeholder.markdown(render_status(detected), unsafe_allow_html=True)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
        
        st.divider()
        
        if best_frame_rgb is not None:
            st.success(f"Highest Threat Level Detected: {highest_confidence:.2f}")
            st.image(best_frame_rgb, caption="Frame with Highest Detection Confidence", use_container_width=True)
            
            if GOOGLE_API_KEY:
                try:
                    with st.spinner("Thinking..."):
                        formatted_text = generate_ai_analysis(best_frame_rgb, GOOGLE_API_KEY)
                        st.markdown(f"""
                        <div class="report-card">
                            <div class="report-header">Results</div>
                            <div class="report-content">
                                {formatted_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(str(e))
            else:
                st.warning("API Key missing")
        else:
            st.markdown("""
            <div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; text-align: center;">
                <h3 style="color: #2e7d32; margin:0;">No Threats Found</h3>
                <p style="color: #666; margin-top: 5px;">The system did not detect any weapons in this video.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        cap.release()
        st.info("Session Ended. Upload a new video to start.")

elif model is None:
    st.error("Model not found. Please ensure 'model/best.pt' exists.")
else:
    st.markdown("""
        <div style='text-align: center; color: #888; padding: 40px;'>
            <p>Upload a video file to begin analysis.</p>
        </div>
    """, unsafe_allow_html=True)
