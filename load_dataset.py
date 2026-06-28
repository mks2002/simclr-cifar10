"""
=========================================================
load_dataset.py

Loads CIFAR-10 and creates

1. Unlabeled dataset (80%)
2. Labeled dataset (20%)
3. Test dataset

=========================================================
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

import config


def load_cifar10():
    """
    Load CIFAR-10 dataset.
    Returns x_train, y_train, x_test, y_test
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize images
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    return x_train, y_train, x_test, y_test


def split_dataset(x_train, y_train):
    """
    Split training set into
    80% unlabeled and   20% labeled
    Returns   x_unlabeled, x_labeled, y_labeled
    """
    np.random.seed(config.RANDOM_SEED)
    indices = np.random.permutation(len(x_train))

    x_train = x_train[indices]
    y_train = y_train[indices]
    split_index = int( len(x_train) * config.UNLABELED_RATIO )

    x_unlabeled = x_train[:split_index]
    x_labeled = x_train[split_index:]
    y_labeled = y_train[split_index:]

    return (
        x_unlabeled,
        x_labeled,
        y_labeled
    )


def create_tf_dataset(images, labels=None, batch_size=128, shuffle=True):
    """
    Creates tf.data Dataset.
    Parameters
    ----------
    images : ndarray
    labels : ndarray or None
    batch_size : int
    shuffle : bool
    """
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(images)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(images),
            seed=config.RANDOM_SEED
        )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch( tf.data.AUTOTUNE)
    return dataset


def get_datasets():
    """
    Main function used by the project.
    Returns : ssl_dataset, finetune_dataset, test_dataset
    """
    x_train, y_train, x_test, y_test = load_cifar10()
    x_unlabeled, x_labeled,y_labeled = split_dataset(x_train, y_train )

    ssl_dataset = create_tf_dataset( x_unlabeled, batch_size=config.SSL_BATCH_SIZE )
    finetune_dataset = create_tf_dataset( x_labeled, y_labeled, batch_size=config.FT_BATCH_SIZE)
    test_dataset = create_tf_dataset(x_test,y_test,batch_size=config.FT_BATCH_SIZE,shuffle=False)

    return (
        ssl_dataset,
        finetune_dataset,
        test_dataset
    )


if __name__ == "__main__":
    ssl_ds, ft_ds, test_ds = get_datasets()
    print()
    print("SSL batches :", len(ssl_ds))
    print("Fine-tuning batches :", len(ft_ds))
    print("Testing batches :", len(test_ds))