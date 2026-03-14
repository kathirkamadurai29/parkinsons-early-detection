"""
Prediction / Inference Module
Load trained CNN-BiLSTM model and predict from .wav file or numpy audio.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extractor import VoiceFeatureExtractor


class ParkinsonsPredictor:
    def __init__(self, model_path='models/best_model.h5'):
        self.model_path = model_path
        self.extractor = VoiceFeatureExtractor()
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
        else:
            print(f"⚠️ Model not found at {self.model_path}. Train the model first.")

    def predict_from_file(self, wav_path):
        """Predict from a .wav audio file."""
        if self.model is None:
            return {'error': 'Model not loaded. Train the model first.'}
        
        features = self.extractor.process_file(wav_path)
        features = np.expand_dims(features, axis=0).astype(np.float32)
        return self._predict(features)

    def predict_from_array(self, audio_array, sr=22050):
        """Predict from raw numpy audio array."""
        if self.model is None:
            return {'error': 'Model not loaded. Train the model first.'}
        
        features = self.extractor.process_numpy_audio(audio_array, sr)
        features = np.expand_dims(features, axis=0).astype(np.float32)
        return self._predict(features)

    def _predict(self, features):
        """Run inference and return structured result."""
        prob = float(self.model.predict(features, verbose=0)[0][0])
        label = "Parkinson's Detected" if prob >= 0.5 else "Healthy"
        risk_level = self._get_risk_level(prob)
        confidence = prob if prob >= 0.5 else (1 - prob)

        return {
            'prediction': label,
            'probability': prob,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_score': round(prob * 100, 2),
            'is_parkinsons': prob >= 0.5,
            'interpretation': self._get_interpretation(prob)
        }

    def _get_risk_level(self, prob):
        if prob < 0.3:
            return 'LOW'
        elif prob < 0.5:
            return 'BORDERLINE'
        elif prob < 0.75:
            return 'MODERATE'
        else:
            return 'HIGH'

    def _get_interpretation(self, prob):
        if prob < 0.3:
            return "Voice patterns appear normal. No significant indicators of Parkinson's disease detected."
        elif prob < 0.5:
            return "Some borderline vocal irregularities detected. Risk is below threshold but monitoring recommended."
        elif prob < 0.75:
            return "Moderate vocal biomarkers consistent with early Parkinson's. Clinical evaluation advised."
        else:
            return "Strong vocal biomarkers associated with Parkinson's disease detected. Immediate medical consultation recommended."

    def batch_predict(self, wav_paths):
        """Predict for a list of .wav files."""
        results = []
        for path in wav_paths:
            r = self.predict_from_file(path)
            r['file'] = os.path.basename(path)
            results.append(r)
        return results


if __name__ == '__main__':
    predictor = ParkinsonsPredictor('models/best_model.h5')
    # Test with a sample file:
    # result = predictor.predict_from_file('sample_voice.wav')
    # print(result)
    print("Predictor ready. Use predictor.predict_from_file('your_voice.wav')")
