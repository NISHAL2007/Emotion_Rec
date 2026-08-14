import cv2
import numpy as np
from config import HAAR_CASCADE_FILE, MEDIAPIPE_MIN_CONFIDENCE

class FaceDetector:
    """
    Robust Face Detector supporting both OpenCV Haar Cascades and MediaPipe Face Detection.
    Handles bounding box clamping and ROI validation for stable real-time operation.
    """
    def __init__(self, backend='haar', min_confidence=MEDIAPIPE_MIN_CONFIDENCE):
        self.backend = backend.lower()
        self.min_confidence = min_confidence
        
        if self.backend == 'haar':
            self.cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
            if self.cascade.empty():
                raise RuntimeError(f"Failed to load Haar cascade from {HAAR_CASCADE_FILE}")
            self.mp_face = None
        elif self.backend in ['mediapipe', 'mp']:
            self.backend = 'mediapipe'
            try:
                import mediapipe as mp
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_face = self.mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=self.min_confidence
                )
            except Exception as e:
                print(f"[Warning] MediaPipe init failed: {e}. Falling back to Haar Cascade.")
                self.backend = 'haar'
                self.cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
                self.mp_face = None
        else:
            raise ValueError(f"Unsupported face detector backend: {backend}")

    def detect_faces(self, frame):
        """
        Detects faces in frame and returns a list of dictionaries containing:
        {'box': (x, y, w, h), 'crop': face_roi}
        """
        h_frame, w_frame = frame.shape[:2]
        detected = []

        if self.backend == 'haar':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for (x, y, w, h) in faces:
                clamped_box, crop = self._clamp_and_crop(frame, x, y, w, h, w_frame, h_frame)
                if crop is not None and crop.size > 0:
                    detected.append({'box': clamped_box, 'crop': crop})
                    
        elif self.backend == 'mediapipe' and self.mp_face is not None:
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

        return detected

    def _clamp_and_crop(self, frame, x, y, w, h, frame_w, frame_h):
        """
        Ensures bounding box stays within frame boundaries to prevent slicing errors.
        """
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)

        if x2 <= x1 or y2 <= y1:
            return (0, 0, 0, 0), None

        clamped_box = (x1, y1, x2 - x1, y2 - y1)
        crop = frame[y1:y2, x1:x2]
        return clamped_box, crop
