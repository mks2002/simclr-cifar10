"""
=========================================================
classifier.py
Classification head for CIFAR-10 fine-tuning
Takes encoder features → outputs class logits
=========================================================
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import config


# =========================================================
# Classification Head
# =========================================================
def build_classifier():
    
    inputs = layers.Input(shape=(config.FEATURE_DIM,))

    # Dense layer
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Output layer (10 classes)
    outputs = layers.Dense( config.NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs, outputs, name="classifier")

    return model