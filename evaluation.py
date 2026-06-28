"""
=========================================================
evaluation.py
Evaluation utilities for SimCLR CIFAR-10 pipeline

Includes:
Accuracy evaluation
Confusion matrix
Classification report
Training curve plotting
=========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import config


# =========================================================
# Evaluate model accuracy
# =========================================================

def evaluate_model(encoder, classifier, test_dataset):

    print("\n📊 Evaluating model...\n")

    y_true = []
    y_pred = []

    for x, y in test_dataset:

        features = encoder(x, training=False)
        preds = classifier(features, training=False)

        preds = tf.argmax(preds, axis=1)

        y_true.extend(y.numpy())
        y_pred.extend(preds.numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred))

    print(f"✅ Test Accuracy: {acc:.4f}")

    return y_true, y_pred


# =========================================================
# Confusion Matrix
# =========================================================
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap( cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# ======================================================
# Classification Report
# =========================================================
def print_classification_report(y_true, y_pred):
    report = classification_report( y_true, y_pred, digits=4)
    print("\n📄 Classification Report:\n")
    print(report)


# =========================================================
# Plot training curves
# =========================================================
def plot_training_curve(loss_list, title="Training Loss"):

    plt.figure(figsize=(8, 5))
    plt.plot(loss_list)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)
    plt.show()