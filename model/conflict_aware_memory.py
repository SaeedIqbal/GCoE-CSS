"""
Implements Conflict-Aware Memory with dynamic sampling.
"""

import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ConflictAwareMemory:
    """Memory buffer that prioritizes high-instability regions."""
    def __init__(self, capacity=100, beta=0.3, gamma_curv=0.1, lambda_rep=0.2, xi=1.5):
        self.capacity = capacity
        self.beta = beta
        self.gamma_curv = gamma_curv
        self.lambda_rep = lambda_rep
        self.xi = xi
        self.memory = []
        self.features = []
        self.sigma = 0.5

    def compute_jacobian(self, model, x):
        x.requires_grad = True
        y = model(x)
        J = []
        for c in range(y.shape[1]):
            grad = torch.autograd.grad(y[:, c].sum(), x, retain_graph=True)[0]
            J.append(grad.norm(dim=1))
        J = torch.stack(J, dim=1)
        return J.norm(p='fro')

    def compute_entropy(self, logits):
        probs = F.softmax(logits / 0.5, dim=1)
        return -(probs * probs.log()).sum(dim=1).mean()

    def compute_curvature(self, logits):
        grad = torch.autograd.grad(logits.sum(), logits, create_graph=True)[0]
        return grad.pow(2).sum().item()

    def compute_redundancy(self, z):
        if len(self.features) == 0:
            return 0.0
        z_flat = z.flatten().cpu().numpy().reshape(1, -1)
        stored = np.array([f.flatten().cpu().numpy() for f in self.features])
        sims = cosine_similarity(z_flat, stored)[0]
        return np.sum(np.exp(-sims**2 / (2 * self.sigma**2)))

    def conflict_potential(self, model, x, z):
        J_norm = self.compute_jacobian(model, x).item()
        entropy = self.compute_entropy(model(x)).item()
        curvature = self.compute_curvature(model(x))
        redundancy = self.compute_redundancy(z)
        psi = J_norm**2 + self.beta * entropy + self.gamma_curv * curvature + self.lambda_rep * redundancy
        return psi

    def update_memory(self, candidates, features, model):
        psi_scores = [self.conflict_potential(model, x, z) for x, z in zip(candidates, features)]
        probs = np.array(psi_scores) ** self.xi
        probs /= probs.sum()
        indices = np.random.choice(len(candidates), size=min(self.capacity, len(candidates)), p=probs, replace=False)
        self.memory = [candidates[i] for i in indices]
        self.features = [features[i] for i in indices]

    def get_replay_loader(self, batch_size=4):
        dataset = torch.utils.data.TensorDataset(
            torch.stack([x for x, _ in self.memory]),
            torch.stack([y for _, y in self.memory])
        )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)