"""
=========================================================
encoder.py
CNN encoder for SimCLR (CIFAR-10)
Output: feature representation vector

=========================================================
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import config


# =========================================================
# Residual Block
# =========================================================

def residual_block(x, filters, stride=1):
    """
    Basic ResNet-style residual block
    """
    shortcut = x
    
    # First conv
    x = layers.Conv2D( filters, kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second conv
    x = layers.Conv2D( filters, kernel_size=3, strides=1, padding="same", use_bias=False )(x)
    x = layers.BatchNormalization()(x)

    # Match dimensions if needed
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D( filters, kernel_size=1, strides=stride, padding="same", use_bias=False )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


# =========================================================
# Encoder Model
# =========================================================

def build_encoder():
    """
    Builds CNN encoder for SimCLR
    """

    inputs = layers.Input(shape=config.INPUT_SHAPE)

    # Initial conv
    x = layers.Conv2D( 64, kernel_size=3, strides=1, padding="same", use_bias=False )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Stage 2
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)

    # Stage 3
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)

    # Stage 4
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)

    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Final feature vector
    outputs = layers.Dense( config.FEATURE_DIM, activation=None )(x)
    model = Model(inputs, outputs, name="encoder")

    return model