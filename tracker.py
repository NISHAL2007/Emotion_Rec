import time
from collections import Counter
from config import EMOTION_LABELS

class SessionStats:
    """
    Tracks session statistics including dominant emotion, FPS metrics,
    and coaching trend summaries.
    """
    def __init__(self):
        self.start_time = time.time()
        self.total_frames = 0
        self.face_detections_count = 0
        self.emotion_counts = Counter({label: 0 for label in EMOTION_LABELS})
        self.coach_summary = None

    def update(self, detected_faces):
        self.total_frames += 1
        self.face_detections_count += len(detected_faces)
        
        for item in detected_faces:
            emotion = item.get('emotion')
            if emotion in self.emotion_counts:
                self.emotion_counts[emotion] += 1

    def set_coach_summary(self, coach_summary_dict):
        """
        Stores coach engine trend summary for final exit report.
        """
        self.coach_summary = coach_summary_dict

    def get_summary(self):
        duration = max(0.1, time.time() - self.start_time)
        avg_fps = self.total_frames / duration if duration > 0 else 0.0
        
        total_emotions = sum(self.emotion_counts.values())
        if total_emotions > 0:
            dominant_emotion, top_count = self.emotion_counts.most_common(1)[0]
            dominant_pct = (top_count / total_emotions) * 100.0
        else:
            dominant_emotion, dominant_pct = "N/A", 0.0

        return {
            'duration_seconds': duration,
            'total_frames': self.total_frames,
            'avg_fps': avg_fps,
            'total_faces_detected': self.face_detections_count,
            'dominant_emotion': dominant_emotion,
            'dominant_percentage': dominant_pct,
            'breakdown': dict(self.emotion_counts),
            'coach_summary': self.coach_summary
        }

    def print_report(self):
        summary = self.get_summary()
        duration_mins = summary['duration_seconds'] / 60.0
        
        print("\n" + "=" * 65)
        print("                SESSION STATISTICS & COACH REPORT                 ")
        print("=" * 65)
        print(f" Total Session Duration : {summary['duration_seconds']:.2f} s ({duration_mins:.2f} mins)")
        print(f" Processed Frames       : {summary['total_frames']}")
        print(f" Average System FPS     : {summary['avg_fps']:.2f} FPS")
        print(f" Total Face Instances   : {summary['total_faces_detected']}")
        print(f" Dominant Emotion       : {summary['dominant_emotion'].upper()} ({summary['dominant_percentage']:.1f}%)")
        
        if summary.get('coach_summary'):
            coach_data = summary['coach_summary']
            print("-" * 65)
            print(" COACHING TREND OBSERVATION:")
            print(f"   - Emotion Shifts Count : {coach_data.get('total_shifts', 0)}")
            print(f"   - Final Observation    : {coach_data.get('final_observation', 'N/A')}")
            
        print("-" * 65)
        print(" EMOTION FREQUENCY BREAKDOWN:")
        
        total_instances = sum(summary['breakdown'].values())
        for emotion, count in summary['breakdown'].items():
            pct = (count / total_instances * 100.0) if total_instances > 0 else 0.0
            bar = "█" * int(pct / 4)
            print(f"   - {emotion:<10}: {count:>5} detections  ({pct:>5.1f}%)  {bar}")
            
        print("=" * 65 + "\n")
