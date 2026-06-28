"""
=========================================================
augmentations.py
SimCLR-style data augmentations for CIFAR-10
Creates TWO augmented views of the same image.

The augmentation pipeline (the methods) is the same, but the random outcomes are different for both views.

=========================================================
"""

import tensorflow as tf
import tensorflow_addons as tfa
import config


# =========================================================
# Random Color Jitter
# =========================================================

def color_jitter(image):
    """
    Randomly changes: brightness, contrast, saturation, and hue of the image
    """
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, 0.6, 1.4)
    image = tf.image.random_saturation(image, 0.6, 1.4)
    image = tf.image.random_hue(image, 0.1)

    return image


# =========================================================
# Random Blur (Simulating SimCLR paper)
# =========================================================
def random_blur(image):
    """
    Applies Gaussian blur using depthwise conv.
    """
    def blur():
        return tfa.image.gaussian_filter2d(
            image,
            filter_shape=3,
            sigma=1.0
        )
    return tf.cond(
        tf.random.uniform([]) < 0.5,
        blur,
        lambda: image
    )


# =========================================================
# Random Crop + Resize
# =========================================================
def random_resized_crop(image):
    """
    Random crop then resize back to original size.
    """
    crop_size = tf.random.uniform(
        shape=[],
        minval=20,
        maxval=32,
        dtype=tf.int32
    )
    image = tf.image.random_crop( image, size=[crop_size, crop_size, 3])
    image = tf.image.resize( image, [config.IMAGE_HEIGHT, config.IMAGE_WIDTH])
    return image


# =========================================================
# Full augmentation pipeline
# =========================================================

def augment(image):
    """
    Applies SimCLR augmentation pipeline.
    """
    image = random_resized_crop(image)               # Random crop + resize
    image = tf.image.random_flip_left_right(image)   # Random flip
    image = color_jitter(image)                      # Color distortions

    # Normalize to [-1, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = (image - 0.5) / 0.5

    return image


# =========================================================
# Two-view generator (MOST IMPORTANT FUNCTION)
# =========================================================

def create_two_views(image):
    """
    For each input image, create:
    view1 = augment(image)
    view2 = augment(image)
    Returns:
    (view1, view2)
    """
    view1 = augment(image)
    view2 = augment(image)

    return view1, view2


# =========================================================
# tf.data wrapper
# =========================================================
def tf_augment(image):
    """
    Wrapper used inside tf.data pipeline
    """
    view1, view2 = tf.py_function(
        func=create_two_views,
        inp=[image],
        Tout=[tf.float32, tf.float32]
    )

    view1.set_shape([config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS])
    view2.set_shape([config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS])
    return view1, view2


# =========================================================
# For supervised fine-tuning (NO augmentation or light aug)
# =========================================================
def supervised_augment(image):
    """
    Light augmentation used in fine-tuning stage.
    """
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_crop(
        tf.image.resize_with_pad(
            image,
            config.IMAGE_HEIGHT + 4,
            config.IMAGE_WIDTH + 4
        ),
        [config.IMAGE_HEIGHT,
         config.IMAGE_WIDTH,
         config.IMAGE_CHANNELS]
    )

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image