"""
Custom evaluation metrics for Geometric Co-Evolution in Continual Semantic Segmentation.
Implements:
- LOCI: Latent-Output Coherence Index
- CARE: Conflict-Aware Replay Efficacy
- MSS: Manifold Stability Score
- SRTR: Semantic Recalibration Trigger Rate
- BT-C: Backward Transfer of Co-Evolution
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform
import warnings

class SegmentationMetric:
    """Base class for segmentation metrics."""
    def reset(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class LOCI(SegmentationMetric):
    """
    Latent-Output Coherence Index (LOCI)
    Measures the alignment between latent feature evolution and logit evolution.
    High LOCI indicates smooth co-evolution.
    """
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.loci_values = []

    def reset(self):
        self.loci_values = []

    def update(self, z_old, z_new, y_old, y_new):
        """
        :param z_old: [B, C, H, W] old latent features
        :param z_new: [B, C, H, W] new latent features
        :param y_old: [B, K, H, W] old logits
        :param y_new: [B, K, H, W] new logits
        """
        with torch.no_grad():
            # Compute latent change
            delta_z = (z_new - z_old).flatten(start_dim=1)  # [B, D]
            delta_z = F.normalize(delta_z, p=2, dim=1)

            # Compute logit change
            delta_y = (y_new - y_old).flatten(start_dim=1)  # [B, K*H*W]
            delta_y = F.normalize(delta_y, p=2, dim=1)

            # Compute coherence: cosine similarity
            coherence = torch.sum(delta_z * delta_y, dim=1)  # [B]
            loci = torch.clamp(coherence, min=0.0, max=1.0)  # Normalize to [0,1]

            self.loci_values.extend(loci.cpu().numpy())

    def compute(self):
        if len(self.loci_values) == 0:
            warnings.warn("LOCI: No values to compute. Returning 0.0")
            return 0.0
        return np.mean(self.loci_values)


class CARE(SegmentationMetric):
    """
    Conflict-Aware Replay Efficacy (CARE)
    Measures how effectively high-instability samples improve performance.
    CARE = IoU_gain_high / IoU_gain_low
    """
    def __init__(self, threshold_quantile=0.8):
        self.threshold_quantile = threshold_quantile
        self.high_perf = []
        self.low_perf = []

    def reset(self):
        self.high_perf = []
        self.low_perf = []

    def update(self, psi_scores, iou_before, iou_after):
        """
        :param psi_scores: [B] conflict potential scores
        :param iou_before: [B] IoU before replay
        :param iou_after: [B] IoU after replay
        """
        with torch.no_grad():
            psi_scores = psi_scores.cpu().numpy()
            iou_before = iou_before.cpu().numpy()
            iou_after = iou_after.cpu().numpy()

            delta_iou = iou_after - iou_before
            threshold = np.quantile(psi_scores, self.threshold_quantile)

            high_mask = psi_scores >= threshold
            low_mask = psi_scores < threshold

            if high_mask.any():
                self.high_perf.append(np.mean(delta_iou[high_mask]))
            if low_mask.any():
                self.low_perf.append(np.mean(delta_iou[low_mask]))

    def compute(self):
        if len(self.high_perf) == 0 or len(self.low_perf) == 0:
            warnings.warn("CARE: Not enough high/low samples. Returning 1.0")
            return 1.0
        care = np.mean(self.high_perf) / (np.mean(self.low_perf) + 1e-8)
        return np.clip(care, 0.5, 3.0)  # Reasonable range


class MSS(SegmentationMetric):
    """
    Manifold Stability Score (MSS)
    Measures feature space stability using MMD between old-class features.
    MSS = exp(-beta * MMD^2)
    """
    def __init__(self, beta=10.0, kernel='rbf', gamma=1.0):
        self.beta = beta
        self.kernel = kernel
        self.gamma = gamma
        self.mss_values = []

    def reset(self):
        self.mss_values = []

    def compute_mmd(self, X, Y):
        """Compute Maximum Mean Discrepancy."""
        X = X.reshape(X.shape[0], -1).cpu().numpy()
        Y = Y.reshape(Y.shape[0], -1).cpu().numpy()
        XX = squareform(pdist(np.concatenate([X, X]), metric=self.kernel))[:len(X), len(X):]
        YY = squareform(pdist(np.concatenate([Y, Y]), metric=self.kernel))[:len(Y), len(Y):]
        XY = squareform(pdist(np.concatenate([X, Y]), metric=self.kernel))[:len(X), len(X):]
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def update(self, features_old_stage, features_new_stage):
        """
        :param features_old_stage: [N, C, H, W] features from old classes at stage t-1
        :param features_new_stage: [N, C, H, W] features from old classes at stage t
        """
        with torch.no_grad():
            mmd = self.compute_mmd(features_old_stage, features_new_stage)
            mss = np.exp(-self.beta * mmd)
            self.mss_values.append(mss)

    def compute(self):
        if len(self.mss_values) == 0:
            return 0.0
        return np.mean(self.mss_values)


class SRTR(SegmentationMetric):
    """
    Semantic Recalibration Trigger Rate (SRTR)
    Fraction of old classes that trigger recalibration.
    """
    def __init__(self, threshold_factor=1.96):
        self.threshold_factor = threshold_factor
        self.history = {}
        self.triggers = []

    def reset(self):
        self.triggers = []

    def update(self, class_id, stability_score):
        """
        :param class_id: int
        :param stability_score: float (S_c value)
        """
        if class_id not in self.history:
            self.history[class_id] = []
        self.history[class_id].append(stability_score)

        if len(self.history[class_id]) < 2:
            self.triggers.append(0)
            return

        mu = np.mean(self.history[class_id][:-1])
        sigma = np.std(self.history[class_id][:-1])
        threshold = mu + self.threshold_factor * sigma

        triggered = 1 if stability_score > threshold else 0
        self.triggers.append(triggered)

    def compute(self):
        if len(self.triggers) == 0:
            return 0.0
        return np.mean(self.triggers)


class BTC(SegmentationMetric):
    """
    Backward Transfer of Co-Evolution (BT-C)
    Measures performance change on old classes after learning new ones.
    BT-C = mean(IoU_final_old - IoU_initial_old)
    """
    def __init__(self):
        self.initial_iou = {}
        self.final_iou = {}

    def reset(self):
        self.initial_iou = {}
        self.final_iou = {}

    def set_initial(self, class_id, iou):
        """Call at the end of stage 1."""
        self.initial_iou[class_id] = iou

    def update(self, class_id, iou):
        """Update final IoU after each stage."""
        self.final_iou[class_id] = iou

    def compute(self):
        if not self.initial_iou or not self.final_iou:
            return 0.0
        btcs = []
        for cls in self.initial_iou:
            if cls in self.final_iou:
                btcs.append(self.final_iou[cls] - self.initial_iou[cls])
        return np.mean(btcs) if btcs else 0.0


__all__ = ['LOCI', 'CARE', 'MSS', 'SRTR', 'BTC']

'''
# trainer/gcoe_trainer.py
"""
GCoE Trainer: Orchestrates Neural ODE, Recalibration, and Conflict-Aware Replay.
"""

import torch
from torch import nn
from model.neural_ode_dynamics import NeuralODECoEvolution
from model.retroactive_recalibration import SemanticRecalibration
from model.conflict_aware_memory import ConflictAwareMemory
from metrics.gcoe_metrics import LOCI, CARE, MSS, SRTR, BTC
from utils import log_info, log_metrics


class GCoETrainer:
    def __init__(self, model, model_old, device, num_classes, old_classes, lambda_kd=1.0):
        self.model = model
        self.model_old = model_old
        self.device = device
        self.num_classes = num_classes
        self.old_classes = old_classes
        self.lambda_kd = lambda_kd

        # Components
        self.ode = NeuralODECoEvolution(D=256, C=num_classes).to(device)
        self.recalibrator = SemanticRecalibration()
        self.memory = ConflictAwareMemory(capacity=100)
        self.t_span = torch.linspace(0, 1, 10)

        # Optimizer
        self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)

        # Metrics
        self.loci = LOCI()
        self.care = CARE()
        self.mss = MSS()
        self.srtr = SRTR()
        self.btc = BTC()

    def train_epoch(self, dataloader, epoch, logger):
        self.model.train()
        total_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            # Neural ODE integration
            z0 = self.model.module.get_backbone_features(x)
            y0 = self.model(x, use_ode=False)
            h0 = torch.cat([z0, y0], dim=1)
            h_final = self.ode(h0, self.t_span)
            z_final, y_final = h_final[:, :-self.num_classes], h_final[:, -self.num_classes:]

            # Loss
            loss_seg = nn.CrossEntropyLoss(ignore_index=255)(y_final, y)
            if self.model_old is not None:
                with torch.no_grad():
                    y_old = self.model_old(x)
                loss_kd = self.kd_loss(y_final, y_old)
            else:
                loss_kd = 0.0

            loss = loss_seg + self.lambda_kd * loss_kd
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()
        log_info(logger, f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")

    def validate(self, dataloader, logger):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x, use_ode=False)
                loss = nn.CrossEntropyLoss(ignore_index=255)(logits, y)
                total_loss += loss.item()
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += (y != 255).sum().item()

        acc = correct / total
        log_info(logger, f"Val Loss: {total_loss / len(dataloader):.4f}, Acc: {acc:.4f}")
        return acc

    def kd_loss(self, y_new, y_old, T=2.0):
        if self.model_old is None:
            return 0.0
        log_prob_new = F.log_softmax(y_new / T, dim=1)
        prob_old = F.softmax(y_old / T, dim=1)
        return F.kl_div(log_prob_new, prob_old, reduction='batchmean') * (T ** 2)

    def run_recalibration(self, dataloader):
        if self.model_old is None:
            return
        for c in range(self.old_classes):
            S_c = self.recalibrator.compute_stability_functional(self.model_old, self.model, dataloader, c)
            self.srtr.update(c, S_c)
            if self.recalibrator.should_recalibrate(c, S_c):
                self.recalibrator.recalibrate_classifier(self.model.module.classifier, S_c)

    def update_memory(self, candidates, features):
        self.memory.update_memory(candidates, features, self.model)

    def log_metrics(self, epoch, logger):
        metrics = {
            'LOCI': self.loci.compute(),
            'CARE': self.care.compute(),
            'MSS': self.mss.compute(),
            'SRTR': self.srtr.compute(),
            'BT-C': self.btc.compute()
        }
        log_metrics(logger, metrics, step=epoch, prefix="val/")
'''