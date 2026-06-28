# SimCLR CIFAR-10

A TensorFlow implementation of the SimCLR self-supervised learning pipeline on CIFAR-10.

The repository includes an end-to-end flow for:

- Loading and splitting CIFAR-10 data
- Generating SimCLR data augmentations
- Training an encoder with contrastive learning
- Fine-tuning a classifier on labeled data
- Evaluating the final model with accuracy, confusion matrix, and classification report

## Repository Structure

- `main.py` - Runs the full pipeline: dataset loading, SSL pretraining, fine-tuning, and evaluation.
- `config.py` - Project configuration, hyperparameters, and directories.
- `load_dataset.py` - Loads CIFAR-10 and prepares unlabeled, labeled, and test `tf.data` datasets.
- `augmentations.py` - SimCLR-style data augmentations and two-view generation.
- `encoder.py` - Builds the CNN encoder used for feature representation.
- `projection_head.py` - Builds the SimCLR projection head.
- `simclr_loss.py` - Implements the NT-Xent contrastive loss.
- `ssl_pretraining.py` - Defines the SimCLR training loop and checkpoint saving.
- `classifier.py` - Builds the fine-tuning classification head.
- `finetuning.py` - Fine-tunes the pretrained encoder with a classifier.
- `evaluation.py` - Evaluates model performance and plots results.
- `utils.py` - Currently empty; reserved for reusable utilities.

## Requirements

This project is designed for Python with the following packages:

- `tensorflow`
- `tensorflow-addons`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

> Recommended Python version: `3.8+`

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the environment:

```bash
# Windows PowerShell
venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat
```

3. Install dependencies:

```bash
pip install tensorflow tensorflow-addons numpy scikit-learn matplotlib seaborn
```

## Usage

Run the full SimCLR pipeline from the repository root:

```bash
python main.py
```

The pipeline performs:

1. CIFAR-10 data loading and splitting
2. Self-supervised SimCLR pretraining
3. Encoder checkpoint saving
4. Supervised fine-tuning
5. Test evaluation with report and confusion matrix

## Configuration

Open `config.py` to modify:

- `SSL_BATCH_SIZE`, `SSL_EPOCHS`, `SSL_LEARNING_RATE`
- `FT_BATCH_SIZE`, `FT_EPOCHS`, `FT_LEARNING_RATE`
- `TEMPERATURE`, `PROJECTION_DIM`, `FEATURE_DIM`
- dataset split ratios and random seed
- checkpoint and result directories

## Notes

- Checkpoints are saved under the `checkpoints` directory.
- The project uses CIFAR-10 directly from `tensorflow.keras.datasets`, so no manual dataset download is required.
- The fine-tuning stage freezes the encoder and trains only the classifier.

## Project Workflow

1. `load_dataset.py` loads CIFAR-10 and creates unlabeled and labeled splits.
2. `augmentations.py` applies SimCLR-style augmentation to generate two views.
3. `encoder.py` and `projection_head.py` define the pretrained representation network.
4. `simclr_loss.py` computes NT-Xent loss for contrastive learning.
5. `ssl_pretraining.py` trains the encoder and saves weights.
6. `classifier.py` and `finetuning.py` build and train the classifier on labeled samples.
7. `evaluation.py` computes accuracy and visualization metrics.

## Contact

If you want to extend this repository, consider adding:

- a `requirements.txt`
- training logs and loss curves
- support for additional datasets
- command-line arguments for configurable runs
