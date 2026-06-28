"""
=========================================================
simclr_loss.py
NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)
for SimCLR self-supervised learning.
This is the correct loss used in SimCLR paper-style setups.
=========================================================
"""

import tensorflow as tf
import config


# =========================================================
# L2 normalization + cosine similarity
# =========================================================
def l2_normalize(x):
    """
    Normalize embeddings to unit length.
    Required for cosine similarity.
    """
    return tf.math.l2_normalize(x, axis=1)


def cosine_similarity_matrix(z):
    """
    Compute full cosine similarity matrix.
    z shape: (2N, D) |  output: (2N, 2N)
    """
    z = l2_normalize(z)
    return tf.matmul(z, z, transpose_b=True)


# =========================================================
# NT-Xent Loss (SimCLR)
# =========================================================
def nt_xent_loss(z_i, z_j, temperature=None):
    """
    Computes SimCLR NT-Xent loss.
    Parameters

    z_i : Tensor (batch_size, dim) Embeddings from view 1
    z_j : Tensor (batch_size, dim) Embeddings from view 2
    temperature : float Softmax temperature scaling
    Returns scalar loss
    """
    if temperature is None:
        temperature = config.TEMPERATURE

    batch_size = tf.shape(z_i)[0]

    # Step 1: Combine both views → (2N, D)
    z = tf.concat([z_i, z_j], axis=0)

    # Step 2: Similarity matrix (cosine similarity)
    z = l2_normalize(z)
    similarity_matrix = tf.matmul(z, z, transpose_b=True)

    # Step 3: Scale by temperature
    logits = similarity_matrix / temperature

    # Step 4: Remove self-similarity
    # (diagonal = very large negative value)
    mask = tf.eye(2 * batch_size)
    logits = logits - mask * 1e9

    # Step 5: Positive pair indices
    # i-th sample in first half ↔ i-th sample in second half
    positive_indices = tf.concat([
        tf.range(batch_size, 2 * batch_size),
        tf.range(0, batch_size)
    ], axis=0)

    # Step 6: Cross-entropy loss
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        positive_indices,
        logits,
        from_logits=True
    )

    # Step 7: Final scalar loss
    return tf.reduce_mean(loss)