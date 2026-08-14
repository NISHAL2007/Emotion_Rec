import os
import sys
import argparse
from config import MODEL_PATH, DEFAULT_SKIP_FRAMES
from train_model import train_emotion_model
from video import VideoProcessor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Facial Emotion Detection System (OpenCV & TensorFlow/Keras)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--source', type=str, default='0',
        help="Video source: webcam index (e.g. 0) or path to video file"
    )
    parser.add_argument(
        '--detector', type=str, default='haar', choices=['haar', 'mediapipe'],
        help="Face detection backend: 'haar' (faster) or 'mediapipe' (more robust)"
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

    # Convert numeric camera source
    source = int(args.source) if args.source.isdigit() else args.source

    # Train model if explicitly requested or if model missing and train directory exists
    if args.train or (not os.path.exists(args.model) and os.path.exists("train")):
        print(f"[Main] Model '{args.model}' not found or --train flag set. Initiating model training...")
        train_emotion_model(epochs=args.epochs, model_save_path=args.model)

    # Initialize and run video processor
    processor = VideoProcessor(
        source=source,
        detector_backend=args.detector,
        model_path=args.model,
        skip_frames=args.skip_frames
    )
    processor.run()

if __name__ == '__main__':
    main()
