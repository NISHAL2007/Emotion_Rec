import time
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np

from detector import FaceDetector
from predictor import EmotionPredictor
from ui import VisualOverlay
from coach import CoachEngine
from tracker import SessionStats
from config import EMOTION_LABELS, EMOTION_COLORS, MODEL_PATH

class EmotionCoachGUI:
    """
    Native Desktop Graphical User Interface (GUI) for Real-Time Facial Emotion Detection & AI Coach.
    Built with Tkinter, OpenCV, and Pillow.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Facial Emotion Detection & AI Coach")
        self.root.geometry("1280x780")
        self.root.configure(bg="#121212")

        # Core Engines
        self.detector = None
        self.predictor = EmotionPredictor(model_path=MODEL_PATH)
        self.overlay = VisualOverlay()
        self.stats = SessionStats()
        self.coach = CoachEngine()
        
        self.cap = None
        self.is_running = False
        self.skip_frames = 2
        self.frame_counter = 0
        self.cached_predictions = []
        self.fps_start = time.time()
        self.fps_count = 0
        self.current_fps = 0.0

        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#121212")
        style.configure("TLabel", background="#121212", foreground="#FFFFFF", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#00E6C3")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 11, "bold"), foreground="#FFB800")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)

    def _build_ui(self):
        # Top Header Bar
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=15, pady=10)
        
        title_lbl = ttk.Label(header_frame, text="🎭 Real-Time Facial Emotion Detection & AI Coach", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT)
        
        sub_lbl = ttk.Label(header_frame, text="GPU Accelerated | MediaPipe & FER-2013 CNN", font=("Segoe UI", 9), foreground="#888888")
        sub_lbl.pack(side=tk.RIGHT)

        # Main Workspace Container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Left Column: Video Feed
        video_container = ttk.Frame(main_container)
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_label = tk.Label(video_container, bg="#1E1E1E", text="Camera Stream Off\nClick 'Start Camera' to launch", font=("Segoe UI", 12), fg="#888888")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Right Column: Side Control & Analytics Dashboard
        side_panel = ttk.Frame(main_container, width=380)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y, expand=False)

        # 1. System Controls Card
        ctrl_frame = tk.LabelFrame(side_panel, text=" ⚙️ System Controls ", bg="#1E1E1E", fg="#00E6C3", font=("Segoe UI", 10, "bold"), bd=1)
        ctrl_frame.pack(fill=tk.X, pady=(0, 10), ipady=5)

        btn_box = ttk.Frame(ctrl_frame)
        btn_box.pack(fill=tk.X, padx=10, pady=5)

        self.btn_start = tk.Button(btn_box, text="▶ Start Camera", bg="#00E6C3", fg="#000000", font=("Segoe UI", 10, "bold"), command=self.start_stream, relief=tk.FLAT)
        self.btn_start.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)

        self.btn_stop = tk.Button(btn_box, text="⏹ Stop Stream", bg="#FF5252", fg="#FFFFFF", font=("Segoe UI", 10, "bold"), command=self.stop_stream, state=tk.DISABLED, relief=tk.FLAT)
        self.btn_stop.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=3)

        # Settings
        cfg_box = ttk.Frame(ctrl_frame)
        cfg_box.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(cfg_box, text="Detector:").grid(row=0, column=0, sticky=tk.W)
        self.detector_var = tk.StringVar(value="mediapipe")
        det_combo = ttk.Combobox(cfg_box, textvariable=self.detector_var, values=["mediapipe", "haar"], state="readonly", width=12)
        det_combo.grid(row=0, column=1, padx=5, pady=3)

        self.mirror_var = tk.BooleanVar(value=True)
        mirror_chk = tk.Checkbutton(cfg_box, text="Mirror Feed", variable=self.mirror_var, bg="#1E1E1E", fg="#FFFFFF", selectcolor="#121212", activebackground="#1E1E1E", activeforeground="#FFFFFF")
        mirror_chk.grid(row=0, column=2, padx=5)

        # 2. AI Coach Cards
        coach_frame = tk.LabelFrame(side_panel, text=" 💡 AI Embodied Coach ", bg="#1E1E1E", fg="#FFB800", font=("Segoe UI", 10, "bold"), bd=1)
        coach_frame.pack(fill=tk.X, pady=(0, 10), ipady=5)

        ttk.Label(coach_frame, text="Micro-Action Suggestion:", style="SubHeader.TLabel").pack(anchor=tk.W, padx=10, pady=(5, 2))
        self.lbl_micro_action = tk.Label(coach_frame, text="Waiting for active stream...", bg="#2A2A2A", fg="#00E6C3", font=("Segoe UI", 10, "bold"), wraplength=340, justify=tk.LEFT, padding=8)
        self.lbl_micro_action.pack(fill=tk.X, padx=10, pady=(0, 8))

        ttk.Label(coach_frame, text="Rolling Window Trend (60s):", style="SubHeader.TLabel").pack(anchor=tk.W, padx=10, pady=(0, 2))
        self.lbl_trend = tk.Label(coach_frame, text="Initializing session trend tracking...", bg="#2A2A2A", fg="#FFB800", font=("Segoe UI", 9.5), wraplength=340, justify=tk.LEFT, padding=8)
        self.lbl_trend.pack(fill=tk.X, padx=10, pady=(0, 5))

        # 3. Real-Time Metrics & Emotion Probabilities
        stats_frame = tk.LabelFrame(side_panel, text=" 📊 Real-Time Metrics ", bg="#1E1E1E", fg="#FFFFFF", font=("Segoe UI", 10, "bold"), bd=1)
        stats_frame.pack(fill=tk.BOTH, expand=True, ipady=5)

        self.lbl_fps = ttk.Label(stats_frame, text="FPS: 0.0  |  Faces Detected: 0", font=("Segoe UI", 10, "bold"), foreground="#00E6C3")
        self.lbl_fps.pack(anchor=tk.W, padx=10, pady=5)

        ttk.Label(stats_frame, text="Live Probabilities:").pack(anchor=tk.W, padx=10, pady=(5, 2))
        
        self.prob_bars = {}
        for emotion in EMOTION_LABELS:
            row = ttk.Frame(stats_frame)
            row.pack(fill=tk.X, padx=10, pady=2)
            
            lbl = ttk.Label(row, text=f"{emotion:<8}", width=10)
            lbl.pack(side=tk.LEFT)
            
            pbar = ttk.Progressbar(row, length=180, mode='determinate')
            pbar.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
            
            val_lbl = ttk.Label(row, text="0%", width=5)
            val_lbl.pack(side=tk.RIGHT)
            
            self.prob_bars[emotion] = (pbar, val_lbl)

    def start_stream(self):
        if self.is_running:
            return
        
        backend = self.detector_var.get()
        self.detector = FaceDetector(backend=backend)
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam. Ensure camera is connected.")
            return

        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        
        self.fps_start = time.time()
        self.fps_count = 0
        self._update_loop()

    def stop_stream(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.video_label.config(image='', text="Camera Stream Off\nClick 'Start Camera' to launch")

    def _update_loop(self):
        if not self.is_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
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

        # Detection & Prediction
        if self.frame_counter % self.skip_frames == 0 or not self.cached_predictions:
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

        # Coach Update
        active_emotions = [p['emotion'] for p in self.cached_predictions]
        micro_text, trend_text = self.coach.update(active_emotions)

        # Overlays
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

        # Display Frame in Tkinter Label
        cv2_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(cv2_rgb)
        
        # Fit to video label container
        v_w = self.video_label.winfo_width()
        v_h = self.video_label.winfo_height()
        if v_w > 10 and v_h > 10:
            img_pil = img_pil.resize((v_w, v_h), Image.Resampling.LANCZOS)

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
