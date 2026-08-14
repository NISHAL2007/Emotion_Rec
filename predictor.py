import os
import cv2
import numpy as np
import tensorflow as tf
from config import EMOTION_LABELS, IMAGE_SIZE, MODEL_PATH

class EmotionPredictor:
    """
    Handles CNN model loading, image ROI preprocessing, and emotion inference.
    Predicts 7 FER-2013 categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
    """
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.labels = EMOTION_LABELS
        self.model = self._load_or_build_model()

    def _load_or_build_model(self):
        if os.path.exists(self.model_path):
            print(f"[EmotionPredictor] Loading model from '{self.model_path}'...")
            try:
                model = tf.keras.models.load_model(self.model_path)
                print("[EmotionPredictor] Model loaded successfully.")
                return model
            except Exception as e:
                print(f"[EmotionPredictor] Error loading model: {e}. Rebuilding standard CNN model structure.")
        
        print(f"[EmotionPredictor] Model file '{self.model_path}' not found or incompatible.")
        print("[EmotionPredictor] Creating fresh untrained CNN architecture...")
        model = self.build_cnn_model()
        return model

    @staticmethod
    def build_cnn_model(input_shape=(48, 48, 1), num_classes=7):
        """
        Builds a standard 4-block Convolutional Neural Network (CNN) tailored for 48x48 FER-2013 dataset.
        """
        model = tf.keras.Sequential([
            # Block 1
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Block 2
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Block 3
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Dense Classifier Block
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def preprocess_face(self, face_bgr):
        """
        Converts BGR face ROI to grayscale, resizes to 48x48, normalizes to [0, 1],
        and reshapes to tensor (1, 48, 48, 1).
        """
        if face_bgr is None or face_bgr.size == 0:
            return None
        
        # Convert to Grayscale if BGR
        if len(face_bgr.shape) == 3 and face_bgr.shape[2] == 3:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_bgr

        # Resize to 48x48
        resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        # Normalize pixel values [0, 255] -> [0.0, 1.0]
        normalized = resized.astype('float32') / 255.0

        # Reshape to (1, 48, 48, 1)
        tensor = np.expand_dims(normalized, axis=-1)
        tensor = np.expand_dims(tensor, axis=0)

        return tensor

    def predict(self, face_bgr):
        """
        Predicts emotion class, confidence percentage, and full probability distribution.
        Returns:
            label (str), confidence (float 0..1), probabilities (dict {label: float})
        """
        tensor = self.preprocess_face(face_bgr)
        if tensor is None:
            return "Neutral", 0.0, {l: 0.0 for l in self.labels}

        preds = self.model.predict(tensor, verbose=0)[0]
        max_idx = int(np.argmax(preds))
        top_label = self.labels[max_idx]
        confidence = float(preds[max_idx])

        prob_dict = {self.labels[i]: float(preds[i]) for i in range(len(self.labels))}
        return top_label, confidence, prob_dict
