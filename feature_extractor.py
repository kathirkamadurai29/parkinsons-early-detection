"""
Feature Extraction for Parkinson's Disease Detection
Extracts Mel-Spectrogram, MFCC, Chroma, ZCR, and Spectral features from voice recordings.
"""

import numpy as np
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')


class VoiceFeatureExtractor:
    def __init__(self, sample_rate=22050, duration=3.0, n_mels=128, n_mfcc=40,
                 hop_length=512, n_fft=2048):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.max_frames = int(np.ceil(sample_rate * duration / hop_length))

    def load_audio(self, file_path):
        """Load and preprocess audio file."""
        y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
        # Pad or trim to fixed duration
        target_length = int(self.sample_rate * self.duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        return y, sr

    def extract_mel_spectrogram(self, y):
        """Extract Mel-Spectrogram (primary 2D input for CNN)."""
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def extract_mfcc(self, y):
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        return np.vstack([mfcc, mfcc_delta, mfcc_delta2])  # (n_mfcc*3, T)

    def extract_chroma(self, y):
        """Extract Chroma features."""
        chroma = librosa.feature.chroma_stft(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return chroma  # (12, T)

    def extract_spectral_features(self, y):
        """Extract spectral contrast and rolloff."""
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=self.sample_rate, hop_length=self.hop_length
        )
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
        return np.vstack([contrast, rolloff, zcr])  # (9, T)

    def extract_voice_quality_features(self, y):
        """Extract jitter-like and shimmer-like features using fundamental frequency."""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([0])
        if len(f0_clean) == 0:
            f0_clean = np.array([0])
        
        jitter = np.std(np.diff(f0_clean)) / (np.mean(f0_clean) + 1e-8)
        hnr = self._compute_hnr(y)
        return jitter, hnr

    def _compute_hnr(self, y):
        """Estimate Harmonics-to-Noise Ratio."""
        autocorr = librosa.autocorrelate(y)
        if len(autocorr) < 2:
            return 0.0
        peak = np.max(autocorr[1:]) if len(autocorr) > 1 else 0
        noise = autocorr[0] - peak + 1e-8
        return 10 * np.log10((peak + 1e-8) / noise)

    def extract_combined_spectrogram(self, y):
        """
        Create stacked 2D feature map for CNN input.
        Shape: (n_features, T, 1) where n_features combines mel + mfcc + chroma + spectral
        """
        mel = self.extract_mel_spectrogram(y)       # (128, T)
        mfcc = self.extract_mfcc(y)                  # (120, T)
        chroma = self.extract_chroma(y)              # (12, T)
        spectral = self.extract_spectral_features(y) # (9, T)

        # Normalize each feature map
        def normalize(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-8)

        mel = normalize(mel)
        mfcc = normalize(mfcc)
        chroma = normalize(chroma)
        spectral = normalize(spectral)

        # Pad/trim time axis
        T = self.max_frames
        def fix_time(x):
            if x.shape[1] < T:
                return np.pad(x, ((0,0),(0, T - x.shape[1])), mode='constant')
            return x[:, :T]

        mel = fix_time(mel)
        mfcc = fix_time(mfcc)
        chroma = fix_time(chroma)
        spectral = fix_time(spectral)

        # Stack: (269, T)
        combined = np.vstack([mel, mfcc, chroma, spectral])
        return combined

    def process_file(self, file_path):
        """Full pipeline: audio file → 2D feature map ready for CNN."""
        y, sr = self.load_audio(file_path)
        features_2d = self.extract_combined_spectrogram(y)
        # Add channel dim: (H, W, 1)
        features_2d = np.expand_dims(features_2d, axis=-1)
        return features_2d

    def process_numpy_audio(self, audio_array, sr):
        """Process raw numpy audio array."""
        # Resample if needed
        if sr != self.sample_rate:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.sample_rate)
        target_length = int(self.sample_rate * self.duration)
        if len(audio_array) < target_length:
            audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
        else:
            audio_array = audio_array[:target_length]
        features_2d = self.extract_combined_spectrogram(audio_array)
        return np.expand_dims(features_2d, axis=-1)
