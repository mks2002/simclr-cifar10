"""
=========================================================
ssl_pretraining.py

Self-Supervised Pretraining using SimCLR (CIFAR-10)

- Encoder + Projection Head training
- NT-Xent loss
- GradientTape training loop
- Checkpoint saving


View 1 ─┐
        ├→ Encoder → Projection Head → z_i ─┐
View 2 ─┘                                  │
                                           ├→ NT-Xent Loss → Backprop
                                           │
                                   Update Encoder + Projection Head
=========================================================
"""

import tensorflow as tf
import os
import config
from simclr_loss import nt_xent_loss


# =========================================================
# SimCLR Pretraining Class
# =========================================================
class SimCLRTrainer:
    def __init__(self, encoder, projection_head, ssl_dataset):

        self.encoder = encoder
        self.projection_head = projection_head
        self.ssl_dataset = ssl_dataset

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.SSL_LEARNING_RATE
        )

        # Checkpoints
        self.checkpoint_dir = config.CHECKPOINT_DIR

        self.checkpoint = tf.train.Checkpoint(
            encoder=self.encoder,
            projection_head=self.projection_head,
            optimizer=self.optimizer
        )

        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.checkpoint_dir,
            max_to_keep=3
        )

        # Loss tracking
        self.train_loss_results = []


    # =========================================================
    # Forward pass
    # =========================================================
    def forward(self, view1, view2):
        """
        Pass both views through encoder + projection head
        """

        # Encoder
        h1 = self.encoder(view1, training=True)
        h2 = self.encoder(view2, training=True)

        # Projection head
        z1 = self.projection_head(h1, training=True)
        z2 = self.projection_head(h2, training=True)

        return z1, z2


    # =========================================================
    # Single training step
    # =========================================================
    @tf.function
    def train_step(self, view1, view2):
        """
        One optimization step
        """

        with tf.GradientTape() as tape:
            z1, z2 = self.forward(view1, view2)
            loss = nt_xent_loss(z1, z2)

        # Compute gradients
        trainable_vars = (
            self.encoder.trainable_variables +
            self.projection_head.trainable_variables
        )

        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(
            zip(gradients, trainable_vars)
        )

        return loss


    # =========================================================
    # Training loop
    # =========================================================
    def train(self, epochs):
        """
        Full SimCLR training loop
        """
        print("\n🚀 Starting SimCLR Pretraining...\n")

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for step, (view1, view2) in enumerate(self.ssl_dataset):
                loss = self.train_step(view1, view2)
                epoch_loss += loss
                num_batches += 1

                if step % 50 == 0:
                    print(
                        f"Epoch {epoch+1}, Step {step}, "
                        f"Loss: {loss.numpy():.4f}"
                    )
            # Average loss
            epoch_loss /= num_batches
            self.train_loss_results.append(epoch_loss.numpy())

            print(
                f"\n📊 Epoch {epoch+1} Completed | "
                f"Loss: {epoch_loss:.4f}\n"
            )
            # Save checkpoint
            self.checkpoint_manager.save()

        print("\n✅ SimCLR Pretraining Completed!\n")
        return self.train_loss_results


    # =========================================================
    # Save final encoder (IMPORTANT)
    # =========================================================
    def save_encoder(self, path=None):
        """
        Save only encoder weights for fine-tuning
        """
        if path is None:
            path = config.PRETRAINED_ENCODER

        self.encoder.save_weights(path)
        print(f"\n💾 Encoder saved at: {path}\n")