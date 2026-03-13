"""
Dataset Preparation & Training Pipeline
Supports: Oxford Parkinson's Voice Dataset + custom .wav files
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pickle
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extractor import VoiceFeatureExtractor
from models.cnn_bilstm import build_cnn_bilstm_model, get_callbacks


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_uci_parkinson_dataset(csv_path='data/parkinsons.data'):
    """
    Load UCI Parkinson's Dataset (pre-extracted features).
    Download from: https://archive.ics.uci.edu/ml/datasets/parkinsons
    Returns X (features), y (labels), feature_names
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ['name', 'status']]
    X = df[feature_cols].values
    y = df['status'].values  # 1=Parkinson's, 0=Healthy
    print(f"UCI Dataset: {X.shape[0]} samples | Parkinson's: {y.sum()} | Healthy: {(y==0).sum()}")
    return X, y, feature_cols


def load_wav_dataset(data_dir, label_map=None, extractor=None):
    """
    Load dataset from .wav files organized as:
        data_dir/
            healthy/  *.wav
            parkinsons/  *.wav
    
    Args:
        data_dir: Root directory
        label_map: {'healthy': 0, 'parkinsons': 1}
        extractor: VoiceFeatureExtractor instance
    """
    if label_map is None:
        label_map = {'healthy': 0, 'parkinsons': 1}
    if extractor is None:
        extractor = VoiceFeatureExtractor()

    X, y, paths = [], [], []
    data_dir = Path(data_dir)

    for class_name, label in label_map.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"⚠️ Directory not found: {class_dir}")
            continue
        wav_files = list(class_dir.glob('*.wav')) + list(class_dir.glob('*.WAV'))
        print(f"Processing {class_name}: {len(wav_files)} files...")
        for wav_path in wav_files:
            try:
                features = extractor.process_file(str(wav_path))
                X.append(features)
                y.append(label)
                paths.append(str(wav_path))
            except Exception as e:
                print(f"  ⚠️ Skipped {wav_path.name}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"\n✅ Loaded {len(X)} samples | Shape: {X.shape}")
    print(f"   Parkinson's: {y.sum()} | Healthy: {(y==0).sum()}")
    return X, y, paths


def generate_synthetic_dataset(n_samples=500, input_shape=(269, 130, 1), seed=42):
    """
    Generate synthetic data for testing the model architecture.
    ONLY for development/testing — replace with real voice data for actual use.
    """
    np.random.seed(seed)
    print("⚠️  Generating SYNTHETIC data for architecture testing...")
    print("   Replace with real Parkinson's voice dataset for actual training!\n")

    X = np.random.randn(n_samples, *input_shape).astype(np.float32)
    # Simulate class differences: Parkinson's has higher variance in features
    y = np.random.randint(0, 2, n_samples)
    X[y == 1] += np.random.randn(*X[y == 1].shape) * 0.5  # More noise for PD
    y = y.astype(np.float32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X, y, input_shape=None, epochs=100, batch_size=32,
                save_dir='models', use_kfold=False, n_folds=5):
    """
    Full training pipeline with optional K-Fold cross-validation.
    
    Args:
        X: Feature array (N, H, W, 1)
        y: Labels (N,)
        input_shape: Override input shape
        epochs: Max training epochs
        batch_size: Training batch size
        save_dir: Model checkpoint directory
        use_kfold: Use stratified K-Fold CV
        n_folds: Number of folds for CV
    
    Returns:
        Trained model, history, evaluation metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    if input_shape is None:
        input_shape = X.shape[1:]

    # Class weights for imbalanced dataset
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")

    if use_kfold:
        return _kfold_training(X, y, input_shape, epochs, batch_size,
                               save_dir, n_folds, class_weight_dict)
    else:
        return _single_split_training(X, y, input_shape, epochs, batch_size,
                                      save_dir, class_weight_dict)


def _single_split_training(X, y, input_shape, epochs, batch_size, save_dir, class_weight_dict):
    """Standard train/val/test split."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=42
    )

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    model = build_cnn_bilstm_model(input_shape)
    model.summary()

    callbacks = get_callbacks(
        model_save_path=os.path.join(save_dir, 'best_model.h5'),
        log_dir=os.path.join(save_dir, 'logs')
    )

    # Data augmentation using tf.data
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train.astype(np.float32)))
        .shuffle(len(X_train))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices((X_val.astype(np.float32), y_val.astype(np.float32)))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("\n🚀 Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test, save_dir)
    save_training_plots(history, save_dir)

    return model, history, metrics


def _kfold_training(X, y, input_shape, epochs, batch_size, save_dir, n_folds, class_weight_dict):
    """Stratified K-Fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*50}")
        print(f" FOLD {fold+1}/{n_folds}")
        print(f"{'='*50}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_cnn_bilstm_model(input_shape)
        fold_save = os.path.join(save_dir, f'fold_{fold+1}')
        os.makedirs(fold_save, exist_ok=True)

        callbacks = get_callbacks(
            model_save_path=os.path.join(fold_save, 'best_model.h5'),
            log_dir=os.path.join(fold_save, 'logs')
        )

        model.fit(
            X_train.astype(np.float32), y_train.astype(np.float32),
            validation_data=(X_val.astype(np.float32), y_val.astype(np.float32)),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, class_weight=class_weight_dict,
            verbose=1
        )

        metrics = evaluate_model(model, X_val, y_val, fold_save, prefix=f'fold_{fold+1}')
        fold_metrics.append(metrics)

    # Average metrics across folds
    avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    std_metrics = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    print("\n📊 K-Fold CV Results:")
    for k in avg_metrics:
        print(f"  {k}: {avg_metrics[k]:.4f} ± {std_metrics[k]:.4f}")

    return model, None, avg_metrics


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, save_dir='.', prefix='test'):
    """Evaluate model and generate comprehensive report."""
    y_pred_prob = model.predict(X_test.astype(np.float32), verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"\n📊 {prefix.upper()} EVALUATION RESULTS")
    print(f"{'─'*40}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', "Parkinson's"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    print(f"  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity         : {specificity:.4f}")

    # Save confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', "Parkinson's"],
                yticklabels=['Healthy', "Parkinson's"])
    plt.title(f'Confusion Matrix — {prefix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_confusion_matrix.png'), dpi=150)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve — Parkinson's Detection")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_roc_curve.png'), dpi=150)
    plt.close()

    metrics = {
        'accuracy': acc, 'auc': auc, 'f1': f1,
        'sensitivity': sensitivity, 'specificity': specificity
    }
    return metrics


def save_training_plots(history, save_dir):
    """Save accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', color='#2196F3')
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='#FF5722')
    axes[0].set_title('Training & Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Acc', color='#4CAF50')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc', color='#FF9800')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("CNN-BiLSTM Training History — Parkinson's Detection", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"\n✅ Training plots saved to {save_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  CNN-BiLSTM Parkinson's Disease Detection — Training")
    print("=" * 60)

    # OPTION A: Train on .wav files
    # X, y, paths = load_wav_dataset('data/voice_samples')

    # OPTION B: Use UCI tabular features (wrap in 2D)
    # X, y, _ = load_uci_parkinson_dataset('data/parkinsons.data')

    # OPTION C: Synthetic data for architecture testing
    INPUT_SHAPE = (269, 130, 1)
    X, y = generate_synthetic_dataset(n_samples=400, input_shape=INPUT_SHAPE)

    model, history, metrics = train_model(
        X, y,
        input_shape=INPUT_SHAPE,
        epochs=50,
        batch_size=16,
        save_dir='models',
        use_kfold=False
    )

    print("\n🎉 Training complete!")
    print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final AUC-ROC:       {metrics['auc']:.4f}")
