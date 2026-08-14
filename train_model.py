import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from predictor import EmotionPredictor
from config import MODEL_PATH, TRAIN_DIR, TEST_DIR, IMAGE_SIZE

def train_emotion_model(epochs=15, batch_size=64, model_save_path=MODEL_PATH):
    """
    Trains CNN emotion classifier on train/ and test/ directory images using GPU acceleration.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[GPU Enabled] Found {len(gpus)} GPU(s): {gpus}")
        print(f"[GPU Enabled] Accelerating training on: NVIDIA GeForce RTX 4050 Laptop GPU")
    else:
        print("[GPU Warning] No GPU detected. Falling back to CPU training.")

    print(f"[Train] Initializing data generators for directory '{TRAIN_DIR}' and '{TEST_DIR}'...")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    if os.path.exists(model_save_path):
        print(f"[Train] Found existing model checkpoint '{model_save_path}'. Loading model for Fine-Tuning...")
        try:
            model = tf.keras.models.load_model(model_save_path, compile=False)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("[Train] Loaded existing weights! Continuing fine-tuning with learning rate = 0.0001")
        except Exception as e:
            print(f"[Train] Error loading model ({e}). Building standard fresh CNN...")
            model = EmotionPredictor.build_cnn_model(input_shape=(48, 48, 1), num_classes=7)
    else:
        print(f"[Train] Building fresh CNN Architecture...")
        model = EmotionPredictor.build_cnn_model(input_shape=(48, 48, 1), num_classes=7)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=6, restore_best_weights=True, verbose=1
        )
    ]

    print(f"[Train] Starting model training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=callbacks
    )

    model.save(model_save_path)
    print(f"[Train] Training complete! Best model saved to '{model_save_path}'.")
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN model on FER-2013 dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--output', type=str, default=MODEL_PATH, help="Path to save output .h5 model")
    args = parser.parse_args()

    train_emotion_model(epochs=args.epochs, batch_size=args.batch_size, model_save_path=args.output)
