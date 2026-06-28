"""
=========================================================
config.py

Configuration file for SimCLR Self-Supervised Learning
on CIFAR-10

Author: Your Name
=========================================================
"""

import os

# =========================================================
# Dataset and Dataset Split
# =========================================================

DATASET_NAME = "CIFAR10"
NUM_CLASSES = 10

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3

INPUT_SHAPE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS
)

UNLABELED_RATIO = 0.80
LABELED_RATIO = 0.20
RANDOM_SEED = 42

# =========================================================
# SSL Training
# =========================================================

SSL_BATCH_SIZE = 256
SSL_EPOCHS = 100
SSL_LEARNING_RATE = 1e-3
TEMPERATURE = 0.5
PROJECTION_DIM = 128

# =========================================================
# Fine-tuning
# =========================================================
FT_BATCH_SIZE = 128
FT_EPOCHS = 30
FT_LEARNING_RATE = 1e-4

# =========================================================
# Encoder
# =========================================================
FEATURE_DIM = 512

# =========================================================
# Saving
# =========================================================

CHECKPOINT_DIR = "checkpoints"

PRETRAINED_ENCODER = os.path.join(
    CHECKPOINT_DIR,
    "encoder.weights.h5"
)

FINAL_MODEL = os.path.join(
    CHECKPOINT_DIR,
    "classifier.weights.h5"
)

# =========================================================
# Plots
# =========================================================

RESULT_DIR = "results"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)