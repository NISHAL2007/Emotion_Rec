import cv2
import numpy as np
from config import EMOTION_COLORS, LOW_LIGHT_THRESHOLD

class VisualOverlay:
    """
    Renders high-quality visual overlays, bounding boxes, confidence progress bars,
    HUD stats, and Embodied-Cognition Coaching prompts.
    """
    def __init__(self, font_scale=0.6, thickness=2):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        self.colors = EMOTION_COLORS

    def draw_face_overlay(self, frame, x, y, w, h, emotion, confidence, probabilities):
        """
        Draws corner-accented bounding box, top emotion banner, and horizontal confidence bar.
        """
        color = self.colors.get(emotion, (180, 180, 180))

        # 1. Corner-Accented Bounding Box
        self._draw_corner_box(frame, x, y, w, h, color)

        # 2. Top Label Banner
        label_text = f"{emotion}: {confidence * 100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, self.font, self.font_scale, self.thickness)
        
        banner_y1 = max(0, y - text_h - 10)
        banner_y2 = y
        cv2.rectangle(frame, (x, banner_y1), (x + text_w + 14, banner_y2), color, cv2.FILLED)
        
        cv2.putText(
            frame, label_text, (x + 7, banner_y2 - 5),
            self.font, self.font_scale, (255, 255, 255), self.thickness, cv2.LINE_AA
        )

        # 3. Horizontal Confidence Progress Bar
        bar_x = x
        bar_y = y + h + 8
        bar_w = w
        bar_h = 10

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), cv2.FILLED)
        fill_w = int(bar_w * max(0.0, min(1.0, confidence)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, cv2.FILLED)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

    def _draw_corner_box(self, frame, x, y, w, h, color):
        line_length = int(min(w, h) * 0.2)
        thick = self.thickness + 1

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

        # Corners
        cv2.line(frame, (x, y), (x + line_length, y), color, thick)
        cv2.line(frame, (x, y), (x, y + line_length), color, thick)

        cv2.line(frame, (x + w, y), (x + w - line_length, y), color, thick)
        cv2.line(frame, (x + w, y), (x + w, y + line_length), color, thick)

        cv2.line(frame, (x, y + h), (x + line_length, y + h), color, thick)
        cv2.line(frame, (x, y + h), (x, y + h - line_length), color, thick)

        cv2.line(frame, (x + w, y + h), (x + w - line_length, y + h), color, thick)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - line_length), color, thick)

    def draw_hud(self, frame, fps, face_count, is_low_light=False):
        """
        Draws top HUD bar displaying FPS, face count, and system status alerts.
        """
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (0, 0), (w, 38), (15, 15, 15), cv2.FILLED)
        cv2.line(frame, (0, 38), (w, 38), (80, 80, 80), 1)

        hud_text = f"EMOTION DETECTOR  |  FPS: {fps:.1f}  |  Faces: {face_count}"
        cv2.putText(frame, hud_text, (15, 24), self.font, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

        if is_low_light:
            cv2.putText(frame, "! LOW LIGHT DETECTED !", (w - 220, 24), self.font, 0.5, (0, 140, 255), 2, cv2.LINE_AA)
        elif face_count == 0:
            cv2.putText(frame, "Searching for faces...", (w - 210, 24), self.font, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    def draw_coaching_overlays(self, frame, micro_action_text, trend_insight_text):
        """
        Renders distinct coaching overlays:
        1. Micro-Action Banner (Bottom of screen)
        2. Session Trend Insight Badge (Top-left below HUD)
        """
        h, w = frame.shape[:2]

        # 1. Micro-Action Banner (Bottom Overlay)
        if micro_action_text:
            banner_h = 40
            banner_y1 = h - banner_h - 10
            banner_y2 = h - 10
            
            # Semi-transparent Dark Cyan/Teal background
            overlay = frame.copy()
            cv2.rectangle(overlay, (15, banner_y1), (w - 15, banner_y2), (40, 30, 20), cv2.FILLED)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            
            cv2.rectangle(frame, (15, banner_y1), (w - 15, banner_y2), (0, 220, 200), 2)

            prompt_label = f"COACH SUGGESTION: {micro_action_text}"
            cv2.putText(
                frame, prompt_label, (30, banner_y2 - 12),
                self.font, 0.55, (255, 255, 255), 1, cv2.LINE_AA
            )

        # 2. Session Trend Insight Badge (Below Top HUD)
        if trend_insight_text:
            badge_y1 = 45
            badge_y2 = 75
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (15, badge_y1), (w - 15, badge_y2), (25, 25, 25), cv2.FILLED)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            
            cv2.rectangle(frame, (15, badge_y1), (w - 15, badge_y2), (180, 120, 40), 1)

            trend_label = f"TREND: {trend_insight_text}"
            cv2.putText(
                frame, trend_label, (30, badge_y2 - 9),
                self.font, 0.5, (0, 230, 255), 1, cv2.LINE_AA
            )

    @staticmethod
    def check_low_light(frame, threshold=LOW_LIGHT_THRESHOLD):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < threshold, float(np.mean(gray))
