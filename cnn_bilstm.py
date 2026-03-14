"""
CNN-BiLSTM Model for Parkinson's Disease Early Detection from Voice
Architecture:
  - 2D Temporal CNN: Extracts local spectro-temporal features from Mel-Spectrogram
  - BiLSTM: Captures long-range temporal dependencies in both directions
  - Dense classifier: Binary output (Healthy / Parkinson's)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
import numpy as np


def build_cnn_bilstm_model(input_shape, num_classes=1, dropout_rate=0.4, l2_reg=1e-4):
    """
    Build CNN-BiLSTM model.
    
    Args:
        input_shape: (H, W, 1) — (n_features, time_frames, 1)
        num_classes: 1 for binary (sigmoid), >2 for softmax
        dropout_rate: Dropout probability
        l2_reg: L2 regularization factor
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='spectrogram_input')
    
    # ─────────────────────────────────────────────────────────
    # BLOCK 1: 2D CNN — Local spectro-temporal feature extraction
    # ─────────────────────────────────────────────────────────
    x = layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg), name='conv2d_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg), name='conv2d_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool_1')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='drop_cnn_1')(x)

    # BLOCK 2: Deeper CNN
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg), name='conv2d_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg), name='conv2d_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool_2')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='drop_cnn_2')(x)

    # BLOCK 3: Deeper CNN
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg), name='conv2d_5')(x)
    x = layers.BatchNormalization(name='bn_5')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), name='pool_3')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='drop_cnn_3')(x)

    # ─────────────────────────────────────────────────────────
    # RESHAPE: CNN output → sequence for BiLSTM
    # (batch, H', W', C) → (batch, W', H'*C)
    # ─────────────────────────────────────────────────────────
    shape = x.shape
    # Merge height and channel dims; keep time (W) as sequence
    x = layers.Permute((2, 1, 3), name='permute')(x)  # (batch, W', H', C)
    x = layers.Reshape((-1, shape[1] * shape[3]), name='reshape_to_sequence')(x)

    # ─────────────────────────────────────────────────────────
    # BLOCK 4: BiLSTM — Temporal context in both directions
    # ─────────────────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=dropout_rate,
                    recurrent_dropout=0.2, name='lstm_1'),
        name='bilstm_1'
    )(x)
    x = layers.Dropout(dropout_rate, name='drop_bilstm_1')(x)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=dropout_rate,
                    recurrent_dropout=0.2, name='lstm_2'),
        name='bilstm_2'
    )(x)
    x = layers.Dropout(dropout_rate, name='drop_bilstm_2')(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=dropout_rate,
                    name='lstm_3'),
        name='bilstm_3'
    )(x)
    x = layers.Dropout(dropout_rate, name='drop_bilstm_3')(x)

    # ─────────────────────────────────────────────────────────
    # BLOCK 5: Classifier Head
    # ─────────────────────────────────────────────────────────
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg), name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_dense_1')(x)
    x = layers.Dropout(dropout_rate, name='drop_dense_1')(x)

    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg), name='dense_2')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='drop_dense_2')(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_BiLSTM_Parkinsons')

    # Compile
    if num_classes == 1:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    return model


def get_callbacks(model_save_path='models/best_model.h5', log_dir='logs'):
    """Get training callbacks."""
    return [
        EarlyStopping(
            monitor='val_auc', patience=15, restore_best_weights=True,
            mode='max', verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7,
            min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path, monitor='val_auc',
            save_best_only=True, mode='max', verbose=1
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]


def get_model_summary(input_shape=(269, 130, 1)):
    """Print model summary."""
    model = build_cnn_bilstm_model(input_shape)
    model.summary()
    return model


if __name__ == '__main__':
    # Quick architecture test
    model = get_model_summary()
    print("\n✅ Model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
