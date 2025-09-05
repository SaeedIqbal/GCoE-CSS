"""
Implements Retroactive Semantic Recalibration using Semantic Stability Functional.
"""

import torch
import torch.nn as nn
import numpy as np

class SemanticRecalibration:
    """Recalibrates classifiers when drift is detected."""
    def __init__(self, eta=0.01, lambda_ortho=0.001, alpha=1.96):
        self.eta = eta
        self.lambda_ortho = lambda_ortho
        self.alpha = alpha  # 95% confidence
        self.history = {}

    def compute_stability_functional(self, model_old, model_new, dataloader, class_idx):
        """Compute S_c = E[||∇log p_k(c|x) - ∇log p_{k-1}(c|x)||²]"""
        S_c_values = []
        model_old.eval()
        model_new.eval()

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.cuda(), y.cuda()
                if (y == class_idx).sum() == 0:
                    continue

                logit_old = model_old(x)
                logit_new = model_new(x)

                log_prob_old = torch.log_softmax(logit_old, dim=1)
                log_prob_new = torch.log_softmax(logit_new, dim=1)

                grad_old = torch.autograd.grad(log_prob_old[:, class_idx].sum(), x, retain_graph=True)[0]
                grad_new = torch.autograd.grad(log_prob_new[:, class_idx].sum(), x, retain_graph=True)[0]

                diff = (grad_new - grad_old).flatten(start_dim=1)
                S_c = torch.norm(diff, p='fro') ** 2
                S_c_values.append(S_c.item())

        return np.mean(S_c_values) if S_c_values else 0.0

    def should_recalibrate(self, class_idx, S_c):
        if class_idx not in self.history or len(self.history[class_idx]) < 2:
            return True
        mu = np.mean(self.history[class_idx])
        sigma = np.std(self.history[class_idx])
        threshold = mu + self.alpha * sigma
        return S_c > threshold

    def orthogonal_regularization(self, classifier):
        W = classifier.weight
        return (W @ W.T - torch.eye(W.size(0), device=W.device)).norm() ** 2

    def recalibrate_classifier(self, classifier, S_c):
        optimizer = torch.optim.SGD([classifier.weight], lr=self.eta)
        reg = self.orthogonal_regularization(classifier)
        loss = S_c + self.lambda_ortho * reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()