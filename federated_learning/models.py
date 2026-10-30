"""
Model definitions shared across centralized, federated and DP training.
"""

from __future__ import annotations

import torch.nn as nn


class LogisticRegressionModel(
    nn.Module
):
    """
    Single linear layer producing one binary-classification logit.
    """

    def __init__(
        self,
        n_features: int,
    ):
        super().__init__()

        self.linear = nn.Linear(
            n_features,
            1,
        )

    def forward(
        self,
        x,
    ):
        return self.linear(
            x
        )