# trainer/gcoe_trainer.py
"""
GCoE-Specific Trainer with:
- Automatic Sample Selection (RL Agent)
- Expert Modules with Shared Information
- Dual-Phase Training
- Compatibility with SOTA continual learning methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import os
from pathlib import Path

# Local imports
from model.segmentation import GCoeSegmentationModel, ExpertModule
from metrics import SegmentationMetrics
from utils import Label2Color, denormalize


class SampleSelectionAgent(nn.Module):
    """
    RL-based Sample Selection Agent (inspired by Replay Master).
    Learns to select memory samples based on diversity and class performance.
    """
    def __init__(self, state_dim=3, hidden_dim=64, action_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Output selection probability
        )

    def forward(self, state):
        return self.network(state)  # [B, 1], probability of selection

    def select_samples(self, dataloader, model, device, num_samples=100):
        """
        Select top-k samples using the agent.
        State: [diversity, accuracy, forgetfulness] per image.
        """
        model.eval()
        scores = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = F.softmax(logits, dim=1).argmax(1)

                # Compute diversity (feature variance)
                features = model.get_backbone_features(x)
                diversity = features.var(dim=[2, 3]).mean().item()

                # Compute accuracy (IoU-like score)
                correct = (pred == y).float()
                accuracy = correct.mean().item()

                # Estimate forgetfulness (class similarity)
                # Simpler: use entropy of class distribution
                class_probs = F.softmax(logits.mean(dim=[2, 3]), dim=1)
                entropy = -(class_probs * class_probs.log()).sum().item()
                forgetfulness = entropy

                state = torch.tensor([[diversity, accuracy, forgetfulness]], device=device)
                score = self(state).item()
                scores.append((score, x.cpu(), y.cpu()))

        # Sort by score and select top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        selected = [(x, y) for _, x, y in scores[:num_samples]]
        return selected


class GCoETrainer:
    """
    Main Trainer for Geometric Co-Evolution Framework.
    Integrates:
    - Neural ODE-based co-evolution
    - Retroactive recalibration
    - Conflict-aware replay
    - Dual-phase training
    """
    def __init__(self, model, model_old, device, num_classes, old_classes,
                 lambda_kd=1.0, use_ode=True, ssf_threshold=0.15, conflict_weight=0.3):
        self.model = model
        self.model_old = model_old
        self.device = device
        self.num_classes = num_classes
        self.old_classes = old_classes
        self.lambda_kd = lambda_kd
        self.use_ode = use_ode
        self.ssf_threshold = ssf_threshold
        self.conflict_weight = conflict_weight

        # Memory buffer
        self.memory = []
        self.features = []

        # RL Agent for sample selection
        self.agent = SampleSelectionAgent().to(device)
        self.optimizer_agent = torch.optim.Adam(self.agent.parameters(), lr=1e-3)

        # Optimizer for segmentation model
        params = [
            {'params': model.module.backbone.parameters(), 'lr': 1e-4},
            {'params': model.module.head.parameters(), 'lr': 1e-3},
        ]
        self.optimizer = torch.optim.SGD(
            params, lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)

        # Metrics
        self.metrics = SegmentationMetrics(num_classes)
        self.label_colorizer = Label2Color(256)  # Max classes

    def knowledge_distillation_loss(self, y_new, y_old, temperature=2.0):
        """Standard KD loss."""
        if self.model_old is None:
            return 0.0
        log_prob_new = F.log_softmax(y_new / temperature, dim=1)
        prob_old = F.softmax(y_old / temperature, dim=1)
        return F.kl_div(log_prob_new, prob_old, reduction='batchmean') * (temperature ** 2)

    def train_epoch(self, dataloader, optimizer, scheduler, epoch, logger):
        """Train for one epoch with dual-phase logic."""
        self.model.train()
        total_loss = 0.0
        loci_accum = 0.0

        # Phase 1: Conventional training
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)

            # Forward
            logits = self.model(x, use_ode=self.use_ode)
            loss_seg = F.cross_entropy(logits, y, ignore_index=255)

            if self.model_old is not None:
                with torch.no_grad():
                    logits_old = self.model_old(x)
                loss_kd = self.knowledge_distillation_loss(logits, logits_old)
            else:
                loss_kd = 0.0

            loss = loss_seg + self.lambda_kd * loss_kd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                logger.info(f"Epoch {epoch}, Iter {i}, Loss: {loss.item():.4f}")

        scheduler.step()
        return total_loss / len(dataloader), loci_accum / len(dataloader)

    def validate(self, dataloader, metrics, logger):
        """Validate model performance."""
        self.model.eval()
        total_loss = 0.0
        metrics.reset()

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(device)
                logits = self.model(x, use_ode=False)
                loss = F.cross_entropy(logits, y, ignore_index=255)
                total_loss += loss.item()

                pred = logits.argmax(1).cpu()
                y = y.cpu()
                metrics.update(pred, y)

        results = metrics.compute()
        return total_loss / len(dataloader), results

    def select_memory_samples(self, dataloader, num_samples=100):
        """Use RL agent to select samples for memory."""
        return self.agent.select_samples(dataloader, self.model, self.device, num_samples)

    def update_memory(self, candidates, model, num_samples=100):
        """Update memory buffer with selected samples."""
        selected = self.select_memory_samples(candidates, model, self.device, num_samples)
        self.memory = selected
        logger.info(f"Memory updated with {len(selected)} samples.")

    def dual_phase_training(self, train_loader, val_loader, epochs_phase1=80, epochs_phase2=20):
        """Dual-phase training: conventional + class-balanced finetuning."""
        logger.info("Phase 1: Conventional Training")
        for epoch in range(epochs_phase1):
            self.train_epoch(train_loader, self.optimizer, self.scheduler, epoch, logger)

        logger.info("Phase 2: Class-Balanced Finetuning")
        # Create balanced dataloader with memory + current data
        balanced_loader = self._create_balanced_loader(train_loader.dataset, self.memory)
        for epoch in range(epochs_phase2):
            self.train_epoch(balanced_loader, self.optimizer, self.scheduler, epoch, logger)

    def _create_balanced_loader(self, current_dataset, memory, batch_size=4):
        """Combine memory and current data for balanced training."""
        from torch.utils.data import ConcatDataset, DataLoader
        memory_dataset = torch.utils.data.TensorDataset(
            torch.stack([x for x, y in memory]),
            torch.stack([y for x, y in memory])
        )
        combined = ConcatDataset([current_dataset, memory_dataset])
        return DataLoader(combined, batch_size=batch_size, shuffle=True)

    def save_checkpoint(self, path):
        """Save model and agent."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load model and agent."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.agent.load_state_dict(ckpt['agent_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")