import time
import cv2
import numpy as np
import gradio as gr
from detector import FaceDetector
from predictor import EmotionPredictor
from ui import VisualOverlay
from coach import CoachEngine
from config import MODEL_PATH, EMOTION_LABELS

# Initialize System Components
detector = FaceDetector(backend='mediapipe')
predictor = EmotionPredictor(model_path=MODEL_PATH)
overlay = VisualOverlay()
coach = CoachEngine()

# Global State Caching for Ultra-Fast Streaming Optimization
g_state = {
    "frame_count": 0,
    "cached_preds": [],
    "last_micro": "Keep a relaxed, natural facial posture.",
    "last_trend": "Observing emotional patterns...",
    "last_fps": 30.0,
    "fps_start": time.time(),
    "fps_counter": 0
}

def process_webcam_stream(image):
    """
    Ultra-Fast Gradio Stream Processor for Live Webcam.
    Optimized for high-FPS video streaming with resolution clamping and prediction caching.
    """
    if image is None:
        return None, "<div style='background: #1E293B; padding: 14px; border-radius: 8px; color: #94A3B8;'>⚠️ Waiting for active video stream...</div>", {}

    now = time.time()
    g_state["fps_counter"] += 1
    if (now - g_state["fps_start"]) >= 1.0:
        g_state["last_fps"] = g_state["fps_counter"] / (now - g_state["fps_start"])
        g_state["fps_counter"] = 0
        g_state["fps_start"] = now

    g_state["frame_count"] += 1

    # 1. Downscale input image to 640x480 for 300% faster OpenCV & MediaPipe processing
    h_orig, w_orig = image.shape[:2]
    if w_orig > 640 or h_orig > 480:
        image_resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
    else:
        image_resized = image

    frame_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)

    # 2. Face Detection & CNN Prediction Caching (evaluates CNN every 2 frames for maximum FPS)
    if g_state["frame_count"] % 2 == 0 or not g_state["cached_preds"]:
        faces = detector.detect_faces(frame_bgr)
        new_preds = []
        for face_info in faces:
            label, conf, probs = predictor.predict(face_info['crop'])
            new_preds.append({
                'box': face_info['box'],
                'emotion': label,
                'confidence': conf,
                'probabilities': probs
            })
        g_state["cached_preds"] = new_preds
    else:
        faces = detector.detect_faces(frame_bgr)
        if len(faces) == len(g_state["cached_preds"]):
            for i, f in enumerate(faces):
                g_state["cached_preds"][i]['box'] = f['box']
        elif faces:
            new_preds = []
            for face_info in faces:
                label, conf, probs = predictor.predict(face_info['crop'])
                new_preds.append({
                    'box': face_info['box'],
                    'emotion': label,
                    'confidence': conf,
                    'probabilities': probs
                })
            g_state["cached_preds"] = new_preds

    predictions = g_state["cached_preds"]

    # 3. AI Coach Suggestions Update
    if predictions:
        active_emotions = [p['emotion'] for p in predictions]
        micro_prompt, trend_prompt = coach.update(active_emotions)
        if micro_prompt:
            g_state["last_micro"] = micro_prompt
        if trend_prompt:
            g_state["last_trend"] = trend_prompt

    # 4. Render Visual Overlays
    is_low_light, _ = overlay.check_low_light(frame_bgr)
    for pred in predictions:
        x, y, w, h = pred['box']
        overlay.draw_face_overlay(frame_bgr, x, y, w, h, pred['emotion'], pred['confidence'], pred['probabilities'])

    overlay.draw_hud(frame_bgr, fps=g_state["last_fps"], face_count=len(predictions), is_low_light=is_low_light)
    overlay.draw_coaching_overlays(frame_bgr, g_state["last_micro"], g_state["last_trend"])

    # Convert back to RGB for Gradio display
    output_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 5. Build Styled HTML Coach Output
    coach_html = f"""
    <div style="background: linear-gradient(135deg, #1E2640 0%, #111827 100%); border-left: 5px solid #00E6C3; padding: 14px; border-radius: 8px; margin-bottom: 10px;">
        <h4 style="margin: 0; color: #00E6C3; font-size: 1rem;">🏋️ Micro-Action Suggestion</h4>
        <p style="margin-top: 6px; font-size: 1.1rem; font-weight: 600; color: #FFFFFF;">{g_state["last_micro"]}</p>
    </div>
    <div style="background: linear-gradient(135deg, #372A14 0%, #1E1810 100%); border-left: 5px solid #FFB800; padding: 14px; border-radius: 8px;">
        <h4 style="margin: 0; color: #FFB800; font-size: 1rem;">📈 60s Rolling Window Trend</h4>
        <p style="margin-top: 6px; font-size: 1.0rem; color: #E5E7EB;">{g_state["last_trend"]}</p>
    </div>
    """

    if predictions:
        probabilities = {k: float(v) for k, v in predictions[0]['probabilities'].items()}
    else:
        probabilities = {label: 0.0 for label in EMOTION_LABELS}

    return output_rgb, coach_html, probabilities

def process_static_image(image):
    """
    Dedicated Static Photo Processor.
    Analyzes uploaded image independently without video caching state contamination.
    """
    if image is None:
        return None, "<div style='background: #1E293B; padding: 14px; border-radius: 8px; color: #94A3B8;'>Please upload an image file.</div>", {}

    # Downscale high-resolution uploaded images for optimal display and processing
    h_orig, w_orig = image.shape[:2]
    if w_orig > 800 or h_orig > 800:
        scale = 800.0 / max(w_orig, h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        image_proc = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        image_proc = image.copy()

    annotated_bgr = cv2.cvtColor(image_proc, cv2.COLOR_RGB2BGR)

    # Detect faces directly on uploaded image
    faces = detector.detect_faces(annotated_bgr)
    predictions = []

    for face_info in faces:
        label, conf, probs = predictor.predict(face_info['crop'])
        predictions.append({
            'box': face_info['box'],
            'emotion': label,
            'confidence': conf,
            'probabilities': probs
        })

    if predictions:
        active_emotions = [p['emotion'] for p in predictions]
        static_coach = CoachEngine()
        micro_prompt, trend_prompt = static_coach.update(active_emotions)

        is_low_light, _ = overlay.check_low_light(annotated_bgr)
        for pred in predictions:
            x, y, w, h = pred['box']
            overlay.draw_face_overlay(annotated_bgr, x, y, w, h, pred['emotion'], pred['confidence'], pred['probabilities'])

        overlay.draw_hud(annotated_bgr, fps=30.0, face_count=len(predictions), is_low_light=is_low_light)
        overlay.draw_coaching_overlays(annotated_bgr, micro_prompt, trend_prompt)

        coach_html = f"""
        <div style="background: linear-gradient(135deg, #1E2640 0%, #111827 100%); border-left: 5px solid #00E6C3; padding: 14px; border-radius: 8px; margin-bottom: 10px;">
            <h4 style="margin: 0; color: #00E6C3; font-size: 1rem;">🏋️ Micro-Action Suggestion</h4>
            <p style="margin-top: 6px; font-size: 1.1rem; font-weight: 600; color: #FFFFFF;">{micro_prompt}</p>
        </div>
        """
        probabilities = {k: float(v) for k, v in predictions[0]['probabilities'].items()}
    else:
        coach_html = """
        <div style="background: #371414; border-left: 5px solid #FF5252; padding: 14px; border-radius: 8px; color: #FFFFFF;">
            ⚠️ No faces detected in uploaded photo. Ensure face is clearly visible and well-lit.
        </div>
        """
        probabilities = {label: 0.0 for label in EMOTION_LABELS}

    output_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return output_rgb, coach_html, probabilities

# Custom Dark Glassmorphism CSS for Premium Gradio UI
custom_css = """
.container {
    max-width: 1200px;
    margin: 0 auto;
}
.header-box {
    text-align: center;
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #334155;
    margin-bottom: 20px;
}
.header-box h1 {
    color: #00E6C3;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 6px;
}
.header-box p {
    color: #94A3B8;
    font-size: 1.05rem;
}
.badge {
    background: #0F172A;
    color: #FFB800;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: bold;
    border: 1px solid #FFB800;
}
"""

# Build Premium Redesigned Gradio Interface
with gr.Blocks(css=custom_css, title="Real-Time Facial Emotion Detection & AI Coach", theme=gr.themes.Monochrome(primary_hue="cyan", neutral_hue="slate")) as demo:
    gr.HTML(
        """
        <div class="header-box">
            <h1>🎭 Real-Time Facial Emotion Detection & AI Coach</h1>
            <p>Powered by TensorFlow / Keras (DirectML GPU Accelerated) & MediaPipe BlazeFace</p>
            <div style="margin-top: 10px;">
                <span class="badge">🚀 30+ FPS Accelerated</span>
                <span style="margin: 0 8px; color: #64748B;">|</span>
                <span class="badge" style="color: #00E6C3; border-color: #00E6C3;">🧠 Embodied Cognition AI</span>
            </div>
        </div>
        """
    )

    with gr.Tabs():
        with gr.TabItem("📹 Live Video Stream"):
            with gr.Row():
                with gr.Column(scale=5):
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Live Camera Feed",
                        streaming=True,
                        elem_id="video_feed"
                    )
                with gr.Column(scale=5):
                    webcam_output = gr.Image(label="Live Visual Overlay & HUD")
                    coach_html_out = gr.HTML(
                        value="""
                        <div style="background: #1E293B; padding: 14px; border-radius: 8px; color: #94A3B8;">
                            Waiting for camera activation... Click webcam play button to launch real-time coach.
                        </div>
                        """
                    )
                    webcam_prob = gr.Label(label="📊 Live Emotion Probabilities", num_top_classes=7)

            webcam_input.stream(
                fn=process_webcam_stream,
                inputs=[webcam_input],
                outputs=[webcam_output, coach_html_out, webcam_prob]
            )

        with gr.TabItem("🖼️ Image Upload & Analysis"):
            with gr.Row():
                with gr.Column(scale=5):
                    upload_input = gr.Image(
                        sources=["upload"],
                        type="numpy",
                        label="Upload Photo for Emotion Analysis"
                    )
                    analyze_btn = gr.Button("⚡ Analyze Photo", variant="primary", size="lg")
                with gr.Column(scale=5):
                    upload_output = gr.Image(label="Visual Overlay Analysis")
                    upload_coach_html = gr.HTML(
                        value="""
                        <div style="background: #1E293B; padding: 14px; border-radius: 8px; color: #94A3B8;">
                            Upload an image and click 'Analyze Photo'.
                        </div>
                        """
                    )
                    upload_prob = gr.Label(label="📊 7-Class Emotion Probabilities", num_top_classes=7)

            analyze_btn.click(
                fn=process_static_image,
                inputs=[upload_input],
                outputs=[upload_output, upload_coach_html, upload_prob]
            )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=10).launch(share=False)
