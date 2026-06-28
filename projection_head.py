"""
=========================================================
projection_head.py
SimCLR projection head (MLP)
Maps encoder features → contrastive embedding space

=========================================================
"""

from tensorflow.keras import layers, Model
import config


# =========================================================
# Projection Head
# =========================================================

def build_projection_head():
    """
    2-layer MLP as used in SimCLR paper
    """
    inputs = layers.Input(shape=(config.FEATURE_DIM,))

    # First layer
    x = layers.Dense(config.FEATURE_DIM, activation="relu" )(inputs)

    # Second layer (final embedding)
    outputs = layers.Dense( config.PROJECTION_DIM, activation=None )(x)
    model = Model(inputs, outputs, name="projection_head")
    return model