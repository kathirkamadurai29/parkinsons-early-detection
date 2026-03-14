"""
Flask Web Application — Parkinson's Disease Early Detection
Provides REST API for voice upload and real-time prediction.
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import sys
import tempfile
import numpy as np
import wave
import struct
import io
import base64
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import ParkinsonsPredictor
from utils.feature_extractor import VoiceFeatureExtractor

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model (will show warning if not trained yet — use synthetic for demo)
predictor = ParkinsonsPredictor('models/best_model.h5')
extractor = VoiceFeatureExtractor()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict/file', methods=['POST'])
def predict_file():
    """Predict from uploaded .wav file."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = predictor.predict_from_file(tmp_path)
        # Also extract visual features for frontend
        import librosa
        y, sr = librosa.load(tmp_path, sr=22050, duration=3.0)
        mel = extractor.extract_mel_spectrogram(y)
        result['mel_spectrogram'] = mel.tolist()
        result['audio_duration'] = len(y) / sr
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route('/api/predict/audio', methods=['POST'])
def predict_audio():
    """Predict from base64 encoded raw PCM audio (from browser recording)."""
    data = request.get_json()
    if not data or 'audio_data' not in data:
        return jsonify({'error': 'No audio data'}), 400

    try:
        audio_b64 = data['audio_data']
        audio_bytes = base64.b64decode(audio_b64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        sr = data.get('sample_rate', 44100)

        result = predictor.predict_from_array(audio_array, sr=sr)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_ready = predictor.model is not None
    return jsonify({
        'status': 'ok',
        'model_loaded': model_ready,
        'model_path': predictor.model_path
    })


@app.route('/api/demo', methods=['GET'])
def demo_predict():
    """Demo prediction with synthetic audio (for testing without trained model)."""
    # Generate synthetic "voice" signal
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    # Simulate voice: fundamental frequency + harmonics + noise
    f0 = 120 + np.random.randn() * 10  # Pitch
    audio = (
        0.5 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.1 * np.sin(2 * np.pi * 3 * f0 * t) +
        0.05 * np.random.randn(len(t))
    ).astype(np.float32)

    try:
        result = predictor.predict_from_array(audio, sr=sr)
    except Exception:
        # If model not loaded, return mock result
        result = {
            'prediction': 'Demo Mode (Model Not Trained)',
            'probability': 0.27,
            'confidence': 0.73,
            'risk_level': 'LOW',
            'risk_score': 27.0,
            'is_parkinsons': False,
            'interpretation': 'This is a demo result. Train the model with real voice data for actual predictions.'
        }

    return jsonify(result)


if __name__ == '__main__':
    print("="*60)
    print("  Parkinson's Detection Web App")
    print("  CNN-BiLSTM Framework")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
