import time
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

from config import EMOTION_COLORS, EMOTION_LABELS, MODEL_PATH
from detector import FaceDetector
from predictor import EmotionPredictor
from ui import VisualOverlay
from coach import CoachEngine
from tracker import SessionStats

class EmotionCoachGUI:
    """
    Native Desktop GUI for Real-Time Facial Emotion Detection & AI Coach.
    Accelerated by TensorFlow (DirectML GPU) & MediaPipe BlazeFace.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Facial Emotion Detection & AI Coach")
        self.root.geometry("1150x700")
        self.root.minsize(950, 580)
        self.root.configure(bg="#121212")

        # Core Backend Engines
        self.detector = None
        self.predictor = None
        self.overlay = VisualOverlay()
        self.stats = SessionStats()
        self.coach = CoachEngine()

        # State Variables
        self.is_running = False
        self.cap = None
        self.frame_counter = 0
        self.cached_predictions = []
        self.fps_start = time.time()
        self.fps_count = 0
        self.current_fps = 0.0

        self._init_styles()
        self._build_ui()

    def _init_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure(".", background="#1E1E1E", foreground="#FFFFFF", fieldbackground="#2A2A2A")
        style.configure("TFrame", background="#121212")
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#00E6C3", background="#121212")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 10, "bold"), foreground="#FFB800", background="#1E1E1E")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)

    def _build_ui(self):
        # Top Header Bar
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=15, pady=8)
        
        title_lbl = ttk.Label(header_frame, text="🎭 Real-Time Facial Emotion Detection & AI Coach", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT)
        
        sub_lbl = ttk.Label(header_frame, text="GPU Accelerated | MediaPipe & FER-2013 CNN", font=("Segoe UI", 9), foreground="#888888", background="#121212")
        sub_lbl.pack(side=tk.RIGHT)

        # Main Workspace Container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Right Column: Side Control & Analytics Dashboard (Fixed Width to prevent expansion)
        side_panel = ttk.Frame(main_container, width=350)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y, expand=False, padx=(10, 0))
        side_panel.pack_propagate(False)

        # Left Column: Video Feed Container
        self.video_container = ttk.Frame(main_container)
        self.video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(
            self.video_container,
            bg="#181818",
            text="Camera Stream Off\nClick '▶ Start Camera' to launch real-time video feed",
            font=("Segoe UI", 12),
            fg="#888888"
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # 1. System Controls Card
        ctrl_frame = tk.LabelFrame(side_panel, text=" ⚙️ System Controls ", bg="#1E1E1E", fg="#00E6C3", font=("Segoe UI", 10, "bold"), bd=1)
        ctrl_frame.pack(fill=tk.X, pady=(0, 10), ipady=4)

        btn_box = ttk.Frame(ctrl_frame)
        btn_box.pack(fill=tk.X, padx=8, pady=4)

        self.btn_start = tk.Button(btn_box, text="▶ Start Camera", bg="#00E6C3", fg="#000000", font=("Segoe UI", 10, "bold"), command=self.start_stream, relief=tk.FLAT, cursor="hand2")
        self.btn_start.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.btn_stop = tk.Button(btn_box, text="⏹ Stop Stream", bg="#FF5252", fg="#FFFFFF", font=("Segoe UI", 10, "bold"), command=self.stop_stream, state=tk.DISABLED, relief=tk.FLAT, cursor="hand2")
        self.btn_stop.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=2)

        # Settings
        cfg_box = ttk.Frame(ctrl_frame)
        cfg_box.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(cfg_box, text="Detector:").grid(row=0, column=0, sticky=tk.W)
        self.detector_var = tk.StringVar(value="mediapipe")
        det_combo = ttk.Combobox(cfg_box, textvariable=self.detector_var, values=["mediapipe", "haar"], state="readonly", width=10)
        det_combo.grid(row=0, column=1, padx=4, pady=2)

        self.mirror_var = tk.BooleanVar(value=True)
        mirror_chk = tk.Checkbutton(cfg_box, text="Mirror Feed", variable=self.mirror_var, bg="#1E1E1E", fg="#FFFFFF", selectcolor="#121212", activebackground="#1E1E1E", activeforeground="#FFFFFF")
        mirror_chk.grid(row=0, column=2, padx=4)

        # 2. AI Coach Cards
        coach_frame = tk.LabelFrame(side_panel, text=" 💡 AI Embodied Coach ", bg="#1E1E1E", fg="#FFB800", font=("Segoe UI", 10, "bold"), bd=1)
        coach_frame.pack(fill=tk.X, pady=(0, 10), ipady=4)

        ttk.Label(coach_frame, text="Micro-Action Suggestion:", style="SubHeader.TLabel").pack(anchor=tk.W, padx=8, pady=(4, 2))
        self.lbl_micro_action = tk.Label(
            coach_frame,
            text="Waiting for active stream...",
            bg="#2A2A2A",
            fg="#00E6C3",
            font=("Segoe UI", 9, "bold"),
            wraplength=310,
            justify=tk.LEFT,
            padx=6, pady=6
        )
        self.lbl_micro_action.pack(fill=tk.X, padx=8, pady=(0, 6))

        ttk.Label(coach_frame, text="Rolling Window Trend (60s):", style="SubHeader.TLabel").pack(anchor=tk.W, padx=8, pady=(0, 2))
        self.lbl_trend = tk.Label(
            coach_frame,
            text="Initializing session trend tracking...",
            bg="#2A2A2A",
            fg="#FFB800",
            font=("Segoe UI", 8),
            wraplength=310,
            justify=tk.LEFT,
            padx=6, pady=6
        )
        self.lbl_trend.pack(fill=tk.X, padx=8, pady=(0, 4))

        # 3. Real-Time Metrics & Emotion Probabilities
        stats_frame = tk.LabelFrame(side_panel, text=" 📊 Real-Time Metrics ", bg="#1E1E1E", fg="#FFFFFF", font=("Segoe UI", 10, "bold"), bd=1)
        stats_frame.pack(fill=tk.BOTH, expand=True, ipady=4)

        self.lbl_fps = ttk.Label(stats_frame, text="FPS: 0.0  |  Faces Detected: 0", font=("Segoe UI", 9, "bold"), foreground="#00E6C3")
        self.lbl_fps.pack(anchor=tk.W, padx=8, pady=4)

        ttk.Label(stats_frame, text="Live Probabilities:").pack(anchor=tk.W, padx=8, pady=(4, 2))
        
        self.prob_bars = {}
        for emotion in EMOTION_LABELS:
            row = ttk.Frame(stats_frame)
            row.pack(fill=tk.X, padx=8, pady=1)
            
            lbl = ttk.Label(row, text=f"{emotion:<8}", width=9)
            lbl.pack(side=tk.LEFT)
            
            pbar = ttk.Progressbar(row, length=150, mode='determinate')
            pbar.pack(side=tk.LEFT, padx=4, expand=True, fill=tk.X)
            
            val_lbl = ttk.Label(row, text="0%", width=4)
            val_lbl.pack(side=tk.RIGHT)
            
            self.prob_bars[emotion] = (pbar, val_lbl)

    def start_stream(self):
        if self.is_running:
            return

        backend = self.detector_var.get()
        self.detector = FaceDetector(backend=backend)
        self.predictor = EmotionPredictor(model_path=MODEL_PATH)
        self.coach = CoachEngine()
        self.stats = SessionStats()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.video_label.config(text="⚠️ Error: Could not access camera (Index 0). Check device permissions.")
            return

        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        
        self.frame_counter = 0
        self.cached_predictions = []
        self.fps_start = time.time()
        self.fps_count = 0
        self.current_fps = 0.0

        self._update_loop()

    def stop_stream(self):
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.video_label.config(image="", text="Camera Stream Off\nClick '▶ Start Camera' to launch real-time video feed")

        if self.coach and hasattr(self, 'stats'):
            self.stats.set_coach_summary(self.coach.get_exit_summary())
            print("\n" + "="*50)
            print("SESSION TERMINATED - SUMMARY REPORT")
            print("="*50)
            summary = self.stats.get_summary()
            for k, v in summary.items():
                print(f"{k}: {v}")

    def _update_loop(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.video_label.config(text="⚠️ Video feed interrupted.")
            self.stop_stream()
            return

        if self.mirror_var.get():
            frame = cv2.flip(frame, 1)

        self.frame_counter += 1
        self.fps_count += 1

        now = time.time()
        if (now - self.fps_start) >= 1.0:
            self.current_fps = self.fps_count / (now - self.fps_start)
            self.fps_count = 0
            self.fps_start = now

        # Face Detection & Inference (Every 2 frames for ultra performance)
        if self.frame_counter % 2 == 0 or not self.cached_predictions:
            detected = self.detector.detect_faces(frame)
            new_preds = []
            for face_info in detected:
                label, conf, probs = self.predictor.predict(face_info['crop'])
                new_preds.append({
                    'box': face_info['box'],
                    'emotion': label,
                    'confidence': conf,
                    'probabilities': probs
                })
            self.cached_predictions = new_preds
        else:
            detected = self.detector.detect_faces(frame)
            if len(detected) == len(self.cached_predictions):
                for i, f in enumerate(detected):
                    self.cached_predictions[i]['box'] = f['box']
            elif detected:
                new_preds = []
                for face_info in detected:
                    label, conf, probs = self.predictor.predict(face_info['crop'])
                    new_preds.append({
                        'box': face_info['box'],
                        'emotion': label,
                        'confidence': conf,
                        'probabilities': probs
                    })
                self.cached_predictions = new_preds
            else:
                self.cached_predictions = []

        self.stats.update(self.cached_predictions)

        # AI Coach Update
        active_emotions = [p['emotion'] for p in self.cached_predictions]
        micro_text, trend_text = self.coach.update(active_emotions)

        # Visual Overlay Rendering
        is_low_light, _ = self.overlay.check_low_light(frame)
        for pred in self.cached_predictions:
            x, y, w, h = pred['box']
            self.overlay.draw_face_overlay(frame, x, y, w, h, pred['emotion'], pred['confidence'], pred['probabilities'])
        
        self.overlay.draw_hud(frame, self.current_fps, len(self.cached_predictions), is_low_light)
        self.overlay.draw_coaching_overlays(frame, micro_text, trend_text)

        # Update GUI Labels & Progress Bars
        self.lbl_micro_action.config(text=micro_text if micro_text else "Keep a relaxed, natural facial posture.")
        self.lbl_trend.config(text=trend_text if trend_text else "Observing emotional patterns...")
        self.lbl_fps.config(text=f"FPS: {self.current_fps:.1f}  |  Faces Detected: {len(self.cached_predictions)}")

        if self.cached_predictions:
            probs = self.cached_predictions[0]['probabilities']
            for emotion, val in probs.items():
                if emotion in self.prob_bars:
                    pbar, val_lbl = self.prob_bars[emotion]
                    pct = int(val * 100)
                    pbar['value'] = pct
                    val_lbl.config(text=f"{pct}%")

        # Convert BGR to RGB for Tkinter Display
        cv2_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calculate fit dimensions using aspect ratio fitting inside container bounds
        container_w = self.video_container.winfo_width()
        container_h = self.video_container.winfo_height()

        if container_w > 50 and container_h > 50:
            frame_h, frame_w = frame.shape[:2]
            aspect = frame_w / frame_h

            target_w = container_w
            target_h = int(target_w / aspect)

            if target_h > container_h:
                target_h = container_h
                target_w = int(target_h * aspect)

            if target_w > 10 and target_h > 10:
                cv2_rgb = cv2.resize(cv2_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        img_pil = Image.fromarray(cv2_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.img_tk = img_tk
        self.video_label.config(image=img_tk, text="")

        # Schedule next update frame (~30 FPS)
        self.root.after(15, self._update_loop)

def main():
    root = tk.Tk()
    app = EmotionCoachGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_stream(), root.destroy()))
    root.mainloop()

if __name__ == '__main__':
    main()
