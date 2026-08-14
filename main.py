import os
import sys
import argparse
from config import MODEL_PATH, DEFAULT_SKIP_FRAMES
from train_model import train_emotion_model
from video import VideoProcessor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Facial Emotion Detection & Embodied-Cognition Coach System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--source', type=str, default='0',
        help="Video source: webcam index (e.g. 0) or path to video file"
    )
    parser.add_argument(
        '--detector', type=str, default='mediapipe', choices=['haar', 'mediapipe'],
        help="Face detection backend: 'mediapipe' (highly accurate) or 'haar'"
    )
    parser.add_argument(
        '--model', type=str, default=MODEL_PATH,
        help="Path to Keras model file (.h5)"
    )
    parser.add_argument(
        '--skip-frames', type=int, default=DEFAULT_SKIP_FRAMES,
        help="Inference frame-skipping factor for FPS optimization"
    )
    parser.add_argument(
        '--no-coach', action='store_true',
        help="Disable embodied-cognition coach engine overlays"
    )
    parser.add_argument(
        '--train', action='store_true',
        help="Train the model on train/ and test/ datasets before launching detection"
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help="Number of epochs if --train is specified"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    if args.train or (not os.path.exists(args.model) and os.path.exists("train")):
        print(f"[Main] Model '{args.model}' not found or --train set. Initiating model training...")
        train_emotion_model(epochs=args.epochs, model_save_path=args.model)

    processor = VideoProcessor(
        source=source,
        detector_backend=args.detector,
        model_path=args.model,
        skip_frames=args.skip_frames,
        enable_coach=not args.no_coach
    )
    processor.run()

if __name__ == '__main__':
    main()
