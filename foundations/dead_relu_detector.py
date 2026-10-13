import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []
        h = x

        with torch.no_grad():
            for layer in model.children():
                h = layer(h)
                if isinstance(layer, nn.ReLU):
                    dead_mask = (h == 0).all(dim=0)
                    dead_fraction = dead_mask.float().mean().item()
                    dead_fractions.append(round(dead_fraction, 4))

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if any(f > 0.5 for f in dead_fractions):
            return 'use_leaky_relu'

        if dead_fractions and dead_fractions[0] > 0.3:
            return 'reinitialize'

        strictly_increasing = all(
            dead_fractions[i] < dead_fractions[i + 1]
            for i in range(len(dead_fractions) - 1)
        )
        if strictly_increasing and dead_fractions and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'

        if dead_fractions and max(dead_fractions) < 0.1:
            return 'healthy'

        return 'healthy'