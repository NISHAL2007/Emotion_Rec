import time
import cv2
import numpy as np
from detector import FaceDetector
from predictor import EmotionPredictor
from ui import VisualOverlay
from tracker import SessionStats
from config import DEFAULT_SKIP_FRAMES, MODEL_PATH

class VideoProcessor:
    """
    Main Real-Time Video Processor integrating Face Detection, CNN Inference,
    Visual Overlays, Frame Skipping, Edge Case Handling, and Session Analytics.
    """
    def __init__(self, source=0, detector_backend='haar', model_path=MODEL_PATH, skip_frames=DEFAULT_SKIP_FRAMES):
        self.source = source
        self.skip_frames = max(1, skip_frames)
        
        print(f"[VideoProcessor] Initializing FaceDetector ({detector_backend})...")
        self.detector = FaceDetector(backend=detector_backend)
        
        print(f"[VideoProcessor] Initializing EmotionPredictor...")
        self.predictor = EmotionPredictor(model_path=model_path)
        
        self.overlay = VisualOverlay()
        self.stats = SessionStats()
        
        # Cache predictions for frame-skipping performance optimization
        self.cached_face_predictions = []
        self.frame_counter = 0

    def run(self):
        """
        Starts the real-time video stream loop.
        """
        print(f"[VideoProcessor] Opening video source: {self.source}")
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            print(f"[Error] Could not open video source {self.source}.")
            print("[Info] If using webcam, ensure camera permissions are granted.")
            return

        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0

        print("\n" + "=" * 55)
        print("  REAL-TIME FACIAL EMOTION DETECTION STARTED  ")
        print("  Press 'q' in the video window to stop and view stats.  ")
        print("=" * 55 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[VideoProcessor] End of stream or empty frame received.")
                    break

                self.frame_counter += 1
                fps_frame_count += 1

                # Calculate real-time FPS
                now = time.time()
                elapsed = now - fps_start_time
                if elapsed >= 1.0:
                    current_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start_time = now

                # 1. Edge Case: Low Lighting Detection
                is_low_light, _ = self.overlay.check_low_light(frame)

                # 2. Face Detection & Emotion Prediction (with Frame Skipping)
                if self.frame_counter % self.skip_frames == 0 or not self.cached_face_predictions:
                    detected = self.detector.detect_faces(frame)
                    new_predictions = []

                    for face_info in detected:
                        box = face_info['box']
                        crop = face_info['crop']

                        # Run CNN Inference
                        label, confidence, prob_dict = self.predictor.predict(crop)

                        new_predictions.append({
                            'box': box,
                            'emotion': label,
                            'confidence': confidence,
                            'probabilities': prob_dict
                        })

                    self.cached_face_predictions = new_predictions
                else:
                    # Frame-skipping: Update face bounding box positions, keep cached emotion predictions
                    detected = self.detector.detect_faces(frame)
                    if len(detected) == len(self.cached_face_predictions):
                        for i, face_info in enumerate(detected):
                            self.cached_face_predictions[i]['box'] = face_info['box']
                    elif detected:
                        # Re-run full inference if face count changed
                        new_predictions = []
                        for face_info in detected:
                            label, confidence, prob_dict = self.predictor.predict(face_info['crop'])
                            new_predictions.append({
                                'box': face_info['box'],
                                'emotion': label,
                                'confidence': confidence,
                                'probabilities': prob_dict
                            })
                        self.cached_face_predictions = new_predictions
                    else:
                        self.cached_face_predictions = []

                # Update session statistics
                self.stats.update(self.cached_face_predictions)

                # 3. Draw Overlays per detected face
                for pred in self.cached_face_predictions:
                    x, y, w, h = pred['box']
                    self.overlay.draw_face_overlay(
                        frame, x, y, w, h,
                        pred['emotion'], pred['confidence'], pred['probabilities']
                    )

                # 4. Draw Top HUD Bar
                self.overlay.draw_hud(frame, current_fps, len(self.cached_face_predictions), is_low_light)

                # 5. Display Video Window
                cv2.imshow("Real-Time Facial Emotion Detection", frame)

                # Quit key listener ('q')
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[VideoProcessor] Exit command 'q' pressed.")
                    break

        except KeyboardInterrupt:
            print("[VideoProcessor] Interrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            # Output session stats on exit
            self.stats.print_report()
