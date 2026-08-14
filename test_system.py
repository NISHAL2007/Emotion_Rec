import os
import cv2
import numpy as np
import tensorflow as tf

def test_imports_and_components():
    print("[Test] Testing system modules...")
    
    # 1. Test Config
    from config import EMOTION_LABELS, EMOTION_COLORS, IMAGE_SIZE, INPUT_SHAPE
    assert len(EMOTION_LABELS) == 7, "EMOTION_LABELS must have 7 classes"
    assert len(EMOTION_COLORS) == 7, "EMOTION_COLORS must cover all 7 classes"
    print("  [OK] config.py validated.")

    # 2. Test Predictor
    from predictor import EmotionPredictor
    model = EmotionPredictor.build_cnn_model(input_shape=INPUT_SHAPE, num_classes=7)
    dummy_input = np.zeros((1, 48, 48, 1), dtype=np.float32)
    pred = model.predict(dummy_input, verbose=0)
    assert pred.shape == (1, 7), f"Expected shape (1, 7), got {pred.shape}"
    assert np.isclose(np.sum(pred), 1.0), "Softmax probabilities must sum to 1.0"
    print("  [OK] predictor.py CNN architecture and tensor shapes validated.")

    # 3. Test Face Crop Preprocessing
    predictor = EmotionPredictor(model_path="dummy.h5")
    dummy_face_bgr = np.ones((100, 120, 3), dtype=np.uint8) * 128
    tensor = predictor.preprocess_face(dummy_face_bgr)
    assert tensor.shape == (1, 48, 48, 1), f"Expected preprocessed shape (1, 48, 48, 1), got {tensor.shape}"
    label, conf, probs = predictor.predict(dummy_face_bgr)
    assert label in EMOTION_LABELS, f"Invalid label predicted: {label}"
    assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"
    print("  [OK] predictor.py preprocessing and inference interface validated.")

    # 4. Test Face Detector Clamping
    from detector import FaceDetector
    detector = FaceDetector(backend='haar')
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    clamped_box, crop = detector._clamp_and_crop(dummy_frame, -10, -10, 100, 100, 640, 480)
    assert clamped_box[0] == 0 and clamped_box[1] == 0, "Box x,y should clamp to >= 0"
    assert crop is not None and crop.shape == (90, 90, 3), f"Unexpected clamped crop shape"
    print("  [OK] detector.py boundary safety clamping validated.")

    # 5. Test Coach Engine
    from coach import CoachEngine
    coach = CoachEngine(debounce_seconds=1.0, window_seconds=10.0, trend_interval=1.0)
    micro_prompt, trend_prompt = coach.update(['Angry'])
    assert len(micro_prompt) > 0, "Micro action prompt should not be empty"
    prompt2, _ = coach.update(['Angry'])
    assert prompt2 == micro_prompt, "Same emotion within debounce interval should return debounced prompt"
    
    summary = coach.get_exit_summary()
    assert 'total_shifts' in summary, "Missing total_shifts in exit summary"
    print("  [OK] coach.py CoachEngine micro-actions, debouncing, & trend engine validated.")

    # 6. Test UI Overlay
    from ui import VisualOverlay
    overlay = VisualOverlay()
    frame_to_draw = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay.draw_face_overlay(frame_to_draw, 50, 50, 100, 100, "Happy", 0.95, probs)
    overlay.draw_hud(frame_to_draw, 30.0, 1, is_low_light=False)
    overlay.draw_coaching_overlays(frame_to_draw, micro_prompt, trend_prompt)
    assert frame_to_draw.sum() > 0, "Overlay should write pixels to frame"
    print("  [OK] ui.py visual overlay & coaching banners validated.")

    # 7. Test Streamlit & Tkinter Desktop GUI Modules
    import app
    assert hasattr(app, 'main'), "app.py must define main()"
    print("  [OK] app.py Streamlit Web Dashboard UI module validated.")

    import gui
    assert hasattr(gui, 'main'), "gui.py must define main()"
    print("  [OK] gui.py Native Tkinter Desktop Application GUI module validated.")

    print("\n[SUCCESS] ALL SYSTEM MODULE, GUIs, & COACH ENGINE TESTS PASSED!\n")

if __name__ == '__main__':
    test_imports_and_components()
