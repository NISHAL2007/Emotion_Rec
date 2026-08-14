import cv2
import numpy as np
from config import HAAR_CASCADE_FILE, MEDIAPIPE_MIN_CONFIDENCE

class FaceDetector:
    """
    Robust Face Detector supporting MediaPipe (Deep Learning BlazeFace) and
    enhanced OpenCV Haar Cascade filtering to eliminate background false positives.
    """
    def __init__(self, backend='mediapipe', min_confidence=MEDIAPIPE_MIN_CONFIDENCE):
        self.backend = backend.lower()
        self.min_confidence = min_confidence
        self.mp_face = None

        if self.backend in ['mediapipe', 'mp', 'auto']:
            try:
                import mediapipe as mp
                if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
                    self.mp_face_detection = mp.solutions.face_detection
                    self.mp_face = self.mp_face_detection.FaceDetection(
                        model_selection=0, min_detection_confidence=self.min_confidence
                    )
                    self.backend = 'mediapipe'
                    print("[FaceDetector] Activated MediaPipe Deep Learning Face Detector (High Accuracy).")
                else:
                    print("[Warning] MediaPipe solutions API not available in installed version. Using Tuned Haar Cascade.")
                    self.backend = 'haar'
            except Exception as e:
                print(f"[Warning] MediaPipe init failed: {e}. Falling back to tuned Haar Cascade.")
                self.backend = 'haar'

        if self.backend == 'haar':
            self.cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
            if self.cascade.empty():
                raise RuntimeError(f"Failed to load Haar cascade from {HAAR_CASCADE_FILE}")
            print("[FaceDetector] Activated Tuned OpenCV Haar Cascade Detector.")

    def detect_faces(self, frame):
        """
        Detects faces in frame and filters out non-face false positives.
        Returns list of dicts: [{'box': (x, y, w, h), 'crop': face_roi}]
        """
        h_frame, w_frame = frame.shape[:2]
        detected = []

        if self.backend == 'mediapipe' and self.mp_face is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x = int(bboxC.xmin * w_frame)
                    y = int(bboxC.ymin * h_frame)
                    w = int(bboxC.width * w_frame)
                    h = int(bboxC.height * h_frame)
                    
                    clamped_box, crop = self._clamp_and_crop(frame, x, y, w, h, w_frame, h_frame)
                    if crop is not None and crop.size > 0:
                        detected.append({'box': clamped_box, 'crop': crop})

        elif self.backend == 'haar':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Increased minNeighbors (8) and minSize (60x60) to eliminate background false positives
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(60, 60)
            )
            for (x, y, w, h) in faces:
                # Aspect Ratio Filter (Human face aspect ratio is approx 0.75 to 1.35)
                aspect_ratio = float(w) / float(h)
                if not (0.75 <= aspect_ratio <= 1.35):
                    continue

                clamped_box, crop = self._clamp_and_crop(frame, x, y, w, h, w_frame, h_frame)
                if crop is not None and crop.size > 0:
                    # Variance Filter to eliminate flat background textures
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
                    if np.std(gray_crop) < 18.0:
                        continue  # Skip low-texture false positives
                    detected.append({'box': clamped_box, 'crop': crop})

        return detected

    def _clamp_and_crop(self, frame, x, y, w, h, frame_w, frame_h):
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)

        if x2 <= x1 or y2 <= y1:
            return (0, 0, 0, 0), None

        clamped_box = (x1, y1, x2 - x1, y2 - y1)
        crop = frame[y1:y2, x1:x2]
        return clamped_box, crop
