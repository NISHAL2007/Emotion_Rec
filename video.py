import time
import cv2
import numpy as np
from detector import FaceDetector
from predictor import EmotionPredictor
from ui import VisualOverlay
from tracker import SessionStats
from coach import CoachEngine
from config import DEFAULT_SKIP_FRAMES, MODEL_PATH

class VideoProcessor:
    """
    Main Real-Time Video Processor integrating Face Detection, CNN Inference,
    Visual Overlays, Embodied Cognition Coach Engine, Frame Skipping, and Analytics.
    """
    def __init__(self, source=0, detector_backend='haar', model_path=MODEL_PATH, skip_frames=DEFAULT_SKIP_FRAMES, enable_coach=True):
        self.source = source
        self.skip_frames = max(1, skip_frames)
        self.enable_coach = enable_coach
        
        print(f"[VideoProcessor] Initializing FaceDetector ({detector_backend})...")
        self.detector = FaceDetector(backend=detector_backend)
        
        print(f"[VideoProcessor] Initializing EmotionPredictor...")
        self.predictor = EmotionPredictor(model_path=model_path)
        
        self.overlay = VisualOverlay()
        self.stats = SessionStats()
        
        if self.enable_coach:
            print("[VideoProcessor] Initializing Embodied Cognition CoachEngine...")
            self.coach = CoachEngine(debounce_seconds=4.5, window_seconds=60.0, trend_interval=10.0)
        else:
            self.coach = None

        self.cached_face_predictions = []
        self.frame_counter = 0

    def run(self):
        print(f"[VideoProcessor] Opening video source: {self.source}")
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            print(f"[Error] Could not open video source {self.source}.")
            print("[Info] If using webcam, ensure camera permissions are granted.")
            return

        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0

        print("\n" + "=" * 60)
        print("  REAL-TIME EMOTION DETECTION & COACH ENGINE STARTED  ")
        print("  Press 'q' in the video window to stop and view stats.  ")
        print("=" * 60 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[VideoProcessor] End of stream or empty frame received.")
                    break

                self.frame_counter += 1
                fps_frame_count += 1

                now = time.time()
                elapsed = now - fps_start_time
                if elapsed >= 1.0:
                    current_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start_time = now

                # 1. Edge Case Check: Low Lighting Detection
                is_low_light, _ = self.overlay.check_low_light(frame)

                # 2. Face Detection & Emotion Prediction (with Frame Skipping)
                if self.frame_counter % self.skip_frames == 0 or not self.cached_face_predictions:
                    detected = self.detector.detect_faces(frame)
                    new_predictions = []

                    for face_info in detected:
                        box = face_info['box']
                        crop = face_info['crop']

                        label, confidence, prob_dict = self.predictor.predict(crop)

                        new_predictions.append({
                            'box': box,
                            'emotion': label,
                            'confidence': confidence,
                            'probabilities': prob_dict
                        })

                    self.cached_face_predictions = new_predictions
                else:
                    detected = self.detector.detect_faces(frame)
                    if len(detected) == len(self.cached_face_predictions):
                        for i, face_info in enumerate(detected):
                            self.cached_face_predictions[i]['box'] = face_info['box']
                    elif detected:
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

                # 3. Update CoachEngine & Generate Overlay Messages
                micro_action_text, trend_text = "", ""
                if self.coach:
                    emotions_in_frame = [p['emotion'] for p in self.cached_face_predictions]
                    micro_action_text, trend_text = self.coach.update(emotions_in_frame)

                # 4. Draw Face Overlays
                for pred in self.cached_face_predictions:
                    x, y, w, h = pred['box']
                    self.overlay.draw_face_overlay(
                        frame, x, y, w, h,
                        pred['emotion'], pred['confidence'], pred['probabilities']
                    )

                # 5. Draw Top HUD Bar & Coaching Banners
                self.overlay.draw_hud(frame, current_fps, len(self.cached_face_predictions), is_low_light)
                
                if self.enable_coach:
                    self.overlay.draw_coaching_overlays(frame, micro_action_text, trend_text)

                # Display Video Frame
                cv2.imshow("Real-Time Facial Emotion Detection & Coach Engine", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[VideoProcessor] Exit key 'q' pressed.")
                    break

        except KeyboardInterrupt:
            print("[VideoProcessor] Interrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

            # Attach coach summary to session stats before reporting
            if self.coach:
                self.stats.set_coach_summary(self.coach.get_exit_summary())
                
            self.stats.print_report()
