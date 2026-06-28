"""
=========================================================
main.py

Full pipeline runner for SimCLR CIFAR-10 project

Steps:
1. Load dataset
2. SSL pretraining
3. Save encoder
4. Fine-tuning
5. Evaluation

=========================================================
"""

from encoder import build_encoder
from projection_head import build_projection_head
from load_dataset import get_datasets
from ssl_pretraining import SimCLRTrainer
from finetuning import FineTuner

from evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    print_classification_report
)

import config

# =========================================================
# MAIN PIPELINE
# =========================================================
def main():

    # 1. Load datasets
    ssl_dataset, train_dataset, test_dataset = get_datasets()
    print("\n📦 Dataset loaded successfully")

    # 2. Build models
    encoder = build_encoder()
    projection_head = build_projection_head()
    print("\n🧠 Models created")

    # 3. SSL Pretraining
    trainer = SimCLRTrainer(
        encoder=encoder,
        projection_head=projection_head,
        ssl_dataset=ssl_dataset
    )
    loss_history = trainer.train(config.SSL_EPOCHS)
    trainer.save_encoder()

    # 4. Fine-tuning
    finetuner = FineTuner(
        encoder=encoder,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )
    finetuner.train(config.FT_EPOCHS)

    # 5. Evaluation
    y_true, y_pred = evaluate_model(
        encoder,
        finetuner.classifier,
        test_dataset
    )
    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    print("\n🎉 Full SimCLR Pipeline Completed!")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()