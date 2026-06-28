"""
=========================================================
finetuning.py

Supervised fine-tuning using labeled CIFAR-10 (20%)
Loads pretrained encoder
Attaches classifier
Trains with supervised loss
Evaluates performance
=========================================================
"""

import tensorflow as tf
import os

import config
from classifier import build_classifier

# =========================================================
# Fine tuning Trainer
# =========================================================
class FineTuner:

    def __init__(self, encoder, train_dataset, test_dataset):

        self.encoder = encoder
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # Freeze encoder initially (VERY IMPORTANT)
        self.encoder.trainable = False

        # Classifier
        self.classifier = build_classifier()

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.FT_LEARNING_RATE
        )
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()



    # Forward pass
    def forward(self, x):

        features = self.encoder(x, training=False)
        predictions = self.classifier(features, training=True)

        return predictions



    # Training step
    @tf.function
    def train_step(self, x, y):

        with tf.GradientTape() as tape:
            preds = self.forward(x)
            loss = self.loss_fn(y, preds)

        trainable_vars = self.classifier.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(
            zip(gradients, trainable_vars)
        )
        self.train_accuracy.update_state(y, preds)

        return loss



    # Evaluation step
    def test_step(self, x, y):

        preds = self.forward(x)
        loss = self.loss_fn(y, preds)
        self.test_accuracy.update_state(y, preds)
        return loss



    # Training loop
    def train(self, epochs):

        print("\n🚀 Starting Fine-tuning...\n")

        for epoch in range(epochs):
            self.train_accuracy.reset_state()
            self.test_accuracy.reset_state()

            train_loss = 0.0
            steps = 0

            # Training
            for x, y in self.train_dataset:

                loss = self.train_step(x, y)
                train_loss += loss
                steps += 1

            # Testing
            for x, y in self.test_dataset:
                self.test_step(x, y)

            train_loss /= steps

            print(
                f"Epoch {epoch+1} | "
                f"Loss: {train_loss:.4f} | "
                f"Train Acc: {self.train_accuracy.result().numpy():.4f} | "
                f"Test Acc: {self.test_accuracy.result().numpy():.4f}"
            )
        print("\n✅ Fine-tuning Completed!\n")