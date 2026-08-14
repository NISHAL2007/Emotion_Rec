import time
import random
from collections import deque, Counter

# Embodied Cognition Micro-Action Suggestions (3 rotating variants per emotion)
MICRO_ACTION_VARIANTS = {
    'Angry': [
        "Unclench your jaw — release facial tension.",
        "Take one slow exhalation right now.",
        "Drop your shoulders away from ears."
    ],
    'Disgust': [
        "Relax your brow and smooth forehead.",
        "Softly exhale and reset your gaze.",
        "Unclench facial muscles for 3 seconds."
    ],
    'Fear': [
        "Name three physical objects around you.",
        "Plant both feet flat on floor.",
        "Exhale slowly for four full counts."
    ],
    'Happy': [
        "Savor this positive moment briefly.",
        "Notice where warmth feels present.",
        "Anchor this good state in mind."
    ],
    'Neutral': [
        "Check posture — keep shoulders relaxed.",
        "Breathe steady — maintain clear focus.",
        "Softly blink to refresh your eyes."
    ],
    'Sad': [
        "Straighten your posture — spine upright.",
        "Lift your gaze to eye level.",
        "Take a full breath into chest."
    ],
    'Surprise': [
        "Breathe gently — let heart rate steady.",
        "Ground your feet flat on floor.",
        "Pause and absorb the moment."
    ]
}

class CoachEngine:
    """
    Decoupled Embodied-Cognition Coaching Engine.
    Provides debounced per-frame micro-action suggestions and rolling-window session trend analysis.
    """
    def __init__(self, debounce_seconds=4.5, window_seconds=60.0, trend_interval=12.0):
        self.micro_actions = MICRO_ACTION_VARIANTS
        self.debounce_seconds = debounce_seconds
        self.window_seconds = window_seconds
        self.trend_interval = trend_interval
        
        # Micro-action state
        self.last_micro_action_time = 0.0
        self.last_emotion = None
        self.current_micro_action = ""
        self.last_variant_index = {emotion: -1 for emotion in MICRO_ACTION_VARIANTS}
        
        # Rolling time window state: stores (timestamp, emotion)
        self.history_window = deque()
        self.last_trend_time = 0.0
        self.current_trend_insight = "Initializing session trend tracking..."
        self.emotion_shift_count = 0
        self.recent_previous_emotion = None

    def update(self, active_emotions):
        """
        Updates coaching state based on current frame detected emotions.
        active_emotions: list of emotion label strings present in frame
        Returns:
            (micro_action_text, trend_insight_text)
        """
        now = time.time()
        
        # Pick primary emotion in frame (if multiple faces, use first or dominant)
        primary_emotion = active_emotions[0] if active_emotions else "Neutral"
        
        # 1. Rolling Window History Update
        self.history_window.append((now, primary_emotion))
        while self.history_window and (now - self.history_window[0][0]) > self.window_seconds:
            self.history_window.popleft()

        # Track emotion shifts
        if self.recent_previous_emotion and primary_emotion != self.recent_previous_emotion:
            self.emotion_shift_count += 1
        self.recent_previous_emotion = primary_emotion

        # 2. Debounced Micro-Action Selection
        time_elapsed = now - self.last_micro_action_time
        emotion_changed = (primary_emotion != self.last_emotion)
        
        if emotion_changed or time_elapsed >= self.debounce_seconds:
            self.current_micro_action = self._get_next_variant(primary_emotion)
            self.last_micro_action_time = now
            self.last_emotion = primary_emotion

        # 3. Rolling Window Session Trend Evaluation
        if (now - self.last_trend_time) >= self.trend_interval:
            self.current_trend_insight = self._evaluate_trends(now)
            self.last_trend_time = now

        return self.current_micro_action, self.current_trend_insight

    def _get_next_variant(self, emotion):
        """
        Selects next micro-action variant ensuring non-repetitive rotation.
        """
        variants = self.micro_actions.get(emotion, self.micro_actions['Neutral'])
        last_idx = self.last_variant_index.get(emotion, -1)
        next_idx = (last_idx + 1) % len(variants)
        self.last_variant_index[emotion] = next_idx
        return variants[next_idx]

    def _evaluate_trends(self, now):
        """
        Analyzes recent 60-second window to detect sustained emotions, rapid shifts, or trends.
        """
        if not self.history_window:
            return "Observing emotional patterns..."

        emotions_in_window = [e for _, e in self.history_window]
        total_samples = len(emotions_in_window)
        counts = Counter(emotions_in_window)
        
        top_emotion, top_count = counts.most_common(1)[0]
        top_ratio = top_count / total_samples

        # Pattern 1: Sustained Single Emotion (> 75% of rolling window)
        if top_ratio >= 0.75 and total_samples > 20:
            if top_emotion == 'Neutral':
                return f"Sustained {top_emotion} for 1 min — small stretch helps reset focus."
            elif top_emotion in ['Angry', 'Sad', 'Fear']:
                return f"Persistent {top_emotion} detected — take a deep grounding breath."
            elif top_emotion == 'Happy':
                return "Consistently positive mood in recent window!"

        # Pattern 2: Frequent Emotion Switching in recent 30 seconds
        recent_30s = [e for t, e in self.history_window if (now - t) <= 30.0]
        recent_shifts = sum(1 for i in range(1, len(recent_30s)) if recent_30s[i] != recent_30s[i-1])
        if recent_shifts >= 4:
            return f"{recent_shifts} emotion shifts in 30s — high engagement or stress."

        # Pattern 3: Dominant Shift / Balanced State
        if top_emotion == 'Happy':
            return "Noticeably positive emotional trend recently."
        
        return f"Recent Trend: Mostly {top_emotion} ({int(top_ratio * 100)}% of last minute)"

    def get_exit_summary(self):
        """
        Returns final coaching trend summary for session exit report.
        """
        if not self.history_window:
            return {
                'total_shifts': self.emotion_shift_count,
                'final_observation': "Session ended before trend analysis completed."
            }

        emotions = [e for _, e in self.history_window]
        counts = Counter(emotions)
        dominant, _ = counts.most_common(1)[0] if counts else ("Neutral", 0)

        observation = f"Dominant state was {dominant} with {self.emotion_shift_count} total shifts."
        return {
            'total_shifts': self.emotion_shift_count,
            'final_observation': observation
        }
