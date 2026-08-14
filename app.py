import time
import cv2
import numpy as np
import streamlit as st
import pandas as pd
from detector import FaceDetector
from predictor import EmotionPredictor
from ui import VisualOverlay
from coach import CoachEngine
from tracker import SessionStats
from config import EMOTION_COLORS, EMOTION_LABELS, MODEL_PATH

# Streamlit Page Config
st.set_page_config(
    page_title="Real-Time Facial Emotion Detection & AI Coach",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism UI)
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stAppHeader {
        background-color: rgba(14, 17, 23, 0.8);
    }
    .coach-card {
        background: linear-gradient(135deg, #1E2640 0%, #111827 100%);
        border-left: 5px solid #00E6C3;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .trend-card {
        background: linear-gradient(135deg, #372A14 0%, #1E1810 100%);
        border-left: 5px solid #FFB800;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🎭 Real-Time Facial Emotion Detection & AI Coach")
    st.caption("Powered by TensorFlow / Keras (DirectML GPU Accelerated) & MediaPipe BlazeFace")

    # Sidebar Controls
    st.sidebar.header("⚙️ System Controls")
    
    detector_choice = st.sidebar.selectbox(
        "Face Detector Backend",
        options=["mediapipe", "haar"],
        index=0,
        help="MediaPipe is recommended for highest accuracy with zero background false positives."
    )
    
    skip_frames = st.sidebar.slider(
        "Frame Skipping (FPS Optimization)",
        min_value=1, max_value=5, value=2,
        help="Process CNN inference every N frames to maintain smooth video playback."
    )

    enable_coach = st.sidebar.checkbox("Enable AI Coach Engine", value=True)
    enable_mirror = st.sidebar.checkbox("Mirror Webcam Feed", value=True)
    camera_index = st.sidebar.number_input("Webcam Index", min_value=0, max_value=5, value=0)
    run_system = st.sidebar.toggle("Start Camera Stream", value=False)

    # Main Layout Columns
    col_video, col_analytics = st.columns([2.2, 1.2])

    with col_video:
        st.subheader("📹 Live Camera Feed & Visual Overlay")
        video_placeholder = st.empty()

    with col_analytics:
        st.subheader("💡 AI Coach & Live Analytics")
        coach_placeholder = st.empty()
        trend_placeholder = st.empty()
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()

    if run_system:
        detector = FaceDetector(backend=detector_choice)
        predictor = EmotionPredictor(model_path=MODEL_PATH)
        overlay = VisualOverlay()
        stats = SessionStats()
        coach = CoachEngine() if enable_coach else None

        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            st.error(f"Could not open webcam index {camera_index}. Please verify device permissions.")
            return

        frame_counter = 0
        cached_predictions = []
        fps_start = time.time()
        fps_count = 0
        current_fps = 0.0
        last_micro_text, last_trend_text = "", ""

        try:
            while run_system:
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.warning("Video stream ended or frame unreadable.")
                    break

                if enable_mirror:
                    frame = cv2.flip(frame, 1)

                frame_counter += 1
                fps_count += 1

                now = time.time()
                if (now - fps_start) >= 1.0:
                    current_fps = fps_count / (now - fps_start)
                    fps_count = 0
                    fps_start = now

                # Face Detection & Inference
                if frame_counter % skip_frames == 0 or not cached_predictions:
                    detected = detector.detect_faces(frame)
                    new_preds = []
                    for face_info in detected:
                        label, conf, probs = predictor.predict(face_info['crop'])
                        new_preds.append({
                            'box': face_info['box'],
                            'emotion': label,
                            'confidence': conf,
                            'probabilities': probs
                        })
                    cached_predictions = new_preds
                else:
                    detected = detector.detect_faces(frame)
                    if len(detected) == len(cached_predictions):
                        for i, f in enumerate(detected):
                            cached_predictions[i]['box'] = f['box']
                    elif detected:
                        new_preds = []
                        for face_info in detected:
                            label, conf, probs = predictor.predict(face_info['crop'])
                            new_preds.append({
                                'box': face_info['box'],
                                'emotion': label,
                                'confidence': conf,
                                'probabilities': probs
                            })
                        cached_predictions = new_preds
                    else:
                        cached_predictions = []

                stats.update(cached_predictions)

                # Coach Update
                micro_text, trend_text = "", ""
                if coach:
                    active_emotions = [p['emotion'] for p in cached_predictions]
                    micro_text, trend_text = coach.update(active_emotions)

                # Draw Visual Overlays
                is_low_light, _ = overlay.check_low_light(frame)
                for pred in cached_predictions:
                    x, y, w, h = pred['box']
                    overlay.draw_face_overlay(frame, x, y, w, h, pred['emotion'], pred['confidence'], pred['probabilities'])
                
                overlay.draw_hud(frame, current_fps, len(cached_predictions), is_low_light)
                if enable_coach:
                    overlay.draw_coaching_overlays(frame, micro_text, trend_text)

                # Convert BGR to RGB for Streamlit display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB")

                # Throttle Streamlit Markdown & Metric updates to 5 Hz (every 6 frames) to prevent UI lag
                if frame_counter % 6 == 0:
                    if enable_coach and (micro_text != last_micro_text or trend_text != last_trend_text):
                        coach_placeholder.markdown(f"""
                        <div class="coach-card">
                            <h4 style="margin:0; color:#00E6C3;">🏋️ Micro-Action Prompt</h4>
                            <p style="font-size: 1.1rem; margin-top:4px; font-weight:600;">{micro_text}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        trend_placeholder.markdown(f"""
                        <div class="trend-card">
                            <h4 style="margin:0; color:#FFB800;">📈 Rolling Window Trend</h4>
                            <p style="font-size: 1.0rem; margin-top:4px;">{trend_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        last_micro_text, last_trend_text = micro_text, trend_text

                    dominant = cached_predictions[0]['emotion'] if cached_predictions else "None"
                    metrics_placeholder.markdown(
                        f"**FPS:** `{current_fps:.1f}` | **Active Faces:** `{len(cached_predictions)}` | **Current Emotion:** `{dominant}`"
                    )

        finally:
            cap.release()
            st.success("Webcam stream stopped.")

            if coach:
                stats.set_coach_summary(coach.get_exit_summary())
            
            st.divider()
            st.subheader("📊 Session Summary Report")
            final_summary = stats.get_summary()
            st.json(final_summary)

    else:
        video_placeholder.info("Toggle 'Start Camera Stream' in the sidebar to launch the system.")

if __name__ == '__main__':
    main()
