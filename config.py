import os

# Emotion classes mapping for FER-2013 dataset (7 classes)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Visual Palette (BGR color scheme for OpenCV overlays)
EMOTION_COLORS = {
    'Angry': (50, 50, 240),      # Vivid Crimson Red
    'Disgust': (40, 160, 60),    # Emerald Green
    'Fear': (180, 50, 200),      # Deep Violet / Purple
    'Happy': (30, 220, 100),     # Vibrant Mint / Lime Green
    'Neutral': (180, 180, 180),  # Soft Slate Gray
    'Sad': (235, 160, 50),       # Ocean Blue / Cyan-Blue
    'Surprise': (10, 215, 255)   # Solar Yellow / Gold
}

# Image Preprocessing Settings
IMAGE_SIZE = (48, 48)
INPUT_SHAPE = (48, 48, 1)

import os
import cv2

# Default File Paths
MODEL_PATH = "emotion_model.h5"
TRAIN_DIR = "train"
TEST_DIR = "test"

# Face Detector Settings
local_cascade = "haarcascade_frontalface_default.xml"
if os.path.exists(local_cascade):
    HAAR_CASCADE_FILE = os.path.abspath(local_cascade)
else:
    HAAR_CASCADE_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

MEDIAPIPE_MIN_CONFIDENCE = 0.5

# Performance and Edge Case Thresholds
LOW_LIGHT_THRESHOLD = 40.0  # Average grayscale intensity threshold for low lighting warning
DEFAULT_SKIP_FRAMES = 2      # Process CNN prediction every N frames to boost FPS
