import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        h = x

        with torch.no_grad():
            for layer in model.children():
                h = layer(h)

                if isinstance(layer, nn.Linear):
                    dead_mask = (h <= 0).all(dim=0)
                    dead_fraction = dead_mask.float().mean().item()
                    stats.append({
                        'mean': round(h.mean().item(), 4),
                        'std': round(h.std().item(), 4),
                        'dead_fraction': round(dead_fraction, 4),
                    })

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()
        y_hat = model(x)
        loss = nn.MSELoss()(y_hat, y)
        loss.backward()

        stats = []
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                g = layer.weight.grad
                stats.append({
                    'mean': round(g.mean().item(), 4),
                    'std': round(g.std().item(), 4),
                    'norm': round(torch.norm(g).item(), 4),
                })

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        for s in activation_stats:
            if s['dead_fraction'] > 0.5:
                return 'dead_neurons'

        for s in gradient_stats:
            if s['norm'] > 1000:
                return 'exploding_gradients'

        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'

        for s in activation_stats:
            if s['std'] < 0.1:
                return 'vanishing_gradients'

        for s in activation_stats:
            if s['std'] > 10.0:
                return 'exploding_gradients'

        return 'healthy'