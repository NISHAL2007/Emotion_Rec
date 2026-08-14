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

def process_emotion_frame(image):
    """
    Gradio Inference Function: Takes image input (RGB numpy array from webcam/upload),
    runs facial emotion detection, visual overlays, and AI coach micro-action prompts.
    """
    if image is None:
        return None, "No image received.", {}

    # Convert RGB from Gradio to BGR for OpenCV processing
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 1. Detect Faces
    faces = detector.detect_faces(frame_bgr)
    predictions = []
    
    for face_info in faces:
        label, conf, probs = predictor.predict(face_info['crop'])
        predictions.append({
            'box': face_info['box'],
            'emotion': label,
            'confidence': conf,
            'probabilities': probs
        })

    # 2. AI Coach Suggestions
    active_emotions = [p['emotion'] for p in predictions]
    micro_prompt, trend_prompt = coach.update(active_emotions)

    # 3. Draw Overlays
    is_low_light, _ = overlay.check_low_light(frame_bgr)
    for pred in predictions:
        x, y, w, h = pred['box']
        overlay.draw_face_overlay(frame_bgr, x, y, w, h, pred['emotion'], pred['confidence'], pred['probabilities'])

    overlay.draw_hud(frame_bgr, fps=30.0, face_count=len(predictions), is_low_light=is_low_light)
    overlay.draw_coaching_overlays(frame_bgr, micro_prompt, trend_prompt)

    # 4. Prepare Output Format
    output_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if predictions:
        probabilities = {k: float(v) for k, v in predictions[0]['probabilities'].items()}
        coach_output = f"🏋️ Micro-Action Prompt:\n{micro_prompt}\n\n📈 60s Rolling Window Trend:\n{trend_prompt}"
    else:
        probabilities = {label: 0.0 for label in EMOTION_LABELS}
        coach_output = "No faces detected in frame. Ensure adequate lighting and face camera."

    return output_rgb, coach_output, probabilities

# Build Gradio Web Interface for Hugging Face Spaces Deployment
with gr.Blocks(title="Real-Time Facial Emotion Detection & AI Coach") as demo:
    gr.Markdown(
        """
        # 🎭 Real-Time Facial Emotion Detection & AI Coach
        ### Powered by TensorFlow/Keras (DirectML GPU) & MediaPipe BlazeFace
        *Production web deployment demo for Hugging Face Spaces.*
        """
    )

    with gr.Tab("📹 Live Webcam Stream"):
        with gr.Row():
            with gr.Column(scale=3):
                webcam_input = gr.Image(
                    sources=["webcam"],
                    type="numpy",
                    label="Live Webcam Input",
                    streaming=True
                )
            with gr.Column(scale=2):
                webcam_output = gr.Image(label="Real-Time Visual Overlay Feed")
                webcam_coach = gr.Textbox(label="💡 AI Embodied Coach", lines=4)
                webcam_prob = gr.Label(label="📊 7-Class Emotion Probabilities", num_top_classes=7)

        webcam_input.stream(
            fn=process_emotion_frame,
            inputs=[webcam_input],
            outputs=[webcam_output, webcam_coach, webcam_prob]
        )

    with gr.Tab("🖼️ Image Upload / Snapshot"):
        with gr.Row():
            with gr.Column(scale=3):
                upload_input = gr.Image(
                    sources=["upload"],
                    type="numpy",
                    label="Upload Image File"
                )
                analyze_btn = gr.Button("⚡ Analyze Image", variant="primary")
            with gr.Column(scale=2):
                upload_output = gr.Image(label="Visual Overlay Analysis")
                upload_coach = gr.Textbox(label="💡 AI Embodied Coach", lines=4)
                upload_prob = gr.Label(label="📊 7-Class Emotion Probabilities", num_top_classes=7)

        analyze_btn.click(
            fn=process_emotion_frame,
            inputs=[upload_input],
            outputs=[upload_output, upload_coach, upload_prob]
        )

if __name__ == "__main__":
    demo.launch(share=False)
