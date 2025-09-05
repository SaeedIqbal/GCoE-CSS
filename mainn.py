# main.py
"""
Entry point for GCoE-CSS training.
"""

import torch
from dataset.voc import VOCIncremental
from dataset.ade20k import ADE20KIncremental
from model.segmentation import GCoeSegmentationModel
from trainer.gcoe_trainer import GCoETrainer
from utils import setup_logger

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("./logs/gcoe_css", rank=0)

    # Dataset
    train_set = VOCIncremental(root="./data/VOC2012", split="train", current_labels=list(range(15)), old_labels=[])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)

    # Model
    model = GCoeSegmentationModel(num_classes=21).to(device)
    model_old = None

    # Trainer
    trainer = GCoETrainer(
        model=model,
        model_old=model_old,
        device=device,
        num_classes=21,
        old_classes=0
    )

    # Training loop
    for epoch in range(80):
        trainer.train_epoch(train_loader, epoch, logger)
        if epoch % 10 == 0:
            trainer.validate(train_loader, logger)
            trainer.log_metrics(epoch, logger)

    logger.info("Training completed.")

if __name__ == "__main__":
    main()