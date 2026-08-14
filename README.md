# Real-Time Facial Emotion Detection System 🎭⚡

A production-quality, modular real-time facial emotion recognition system built with **Python**, **OpenCV**, and **TensorFlow / Keras**. Accelerated with **DirectML GPU support** (NVIDIA GeForce RTX / AMD / Intel GPUs).

Predicts 7 FER-2013 facial emotion classes: **Angry**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**.

---

## 🌟 Key Features

- **Modular Architecture**: Separated into dedicated classes (`FaceDetector`, `EmotionPredictor`, `VisualOverlay`, `VideoProcessor`, `SessionStats`).
- **Flexible Face Detection**: Supports both **OpenCV Haar Cascades** and **MediaPipe Face Detection** with coordinate safety clamping to handle edge boundaries gracefully.
- **Polished Visual Overlay**:
  - Color-coded bounding boxes per emotion (e.g. Green for Happy, Crimson Red for Angry, Soft Blue for Sad).
  - Corner-accented bounding box aesthetics.
  - Horizontal confidence progress bar under each face.
  - Real-time top HUD displaying system FPS, face count, and status alerts.
- **Edge Case Resilience**: Automatic low-lighting warning indicator and no-face detection fallback.
- **FPS Optimization**: Configurable frame-skipping algorithm to maintain smooth real-time video frames while optimizing CNN inference.
- **Session Analytics**: Prints a detailed summary report upon pressing `'q'` (dominant emotion detected, total frames, average FPS, and emotion frequency breakdown).
- **GPU Acceleration**: Built-in support for TensorFlow GPU / DirectML training and real-time execution.

---

## 🛠️ Project Structure

```
hackorbit/
├── config.py             # Global constants, color palettes, threshold settings
├── detector.py           # FaceDetector class (Haar Cascade & MediaPipe with boundary safety)
├── predictor.py          # EmotionPredictor class (Keras CNN inference & 48x48 preprocessing)
├── ui.py                 # VisualOverlay class (Color-coded boxes, progress bars, HUD)
├── tracker.py            # SessionStats class (Dominant emotion calculation & reports)
├── video.py              # VideoProcessor class (Real-time video loop & controls)
├── train_model.py        # FER-2013 GPU model training script
├── main.py               # Application entry point CLI
└── test_system.py        # Automated test suite
```

---

## 🚀 Quick Start

### 1. Requirements & Setup

Ensure Python 3.8+ and install required dependencies:

```bash
pip install opencv-python tensorflow tensorflow-directml mediapipe numpy pillow
```

### 2. Run Real-Time Emotion Detection

Launch the real-time webcam detector:

```bash
python main.py
```

#### CLI Options:

- **Use MediaPipe Face Detector**:
  ```bash
  python main.py --detector mediapipe
  ```
- **Custom Video File / Stream Source**:
  ```bash
  python main.py --source path/to/video.mp4
  ```
- **Adjust Frame Skipping Factor**:
  ```bash
  python main.py --skip-frames 1
  ```

### 3. Model Training (GPU Accelerated)

Train or fine-tune the CNN on FER-2013 dataset subdirectories (`train/`, `test/`):

```bash
python train_model.py --epochs 15
```

---

## ⌨️ Controls & Keyboard Shortcuts

- **`q` or `Esc`**: Exit the video window cleanly and output the session analytics report in console.

---

## 📊 Session Report Output Example

```
============================================================
                SESSION STATISTICS REPORT                 
============================================================
 Total Session Duration : 45.20 s (0.75 mins)
 Processed Frames       : 1356
 Average System FPS     : 30.00 FPS
 Total Face Instances   : 1350
 Dominant Emotion       : HAPPY (84.5%)
------------------------------------------------------------
 EMOTION FREQUENCY BREAKDOWN:
   - Happy     :  1141 detections  ( 84.5%)  ████████████████████
   - Neutral   :   152 detections  ( 11.3%)  ██
   - Surprise  :    57 detections  (  4.2%)  █
============================================================
```
