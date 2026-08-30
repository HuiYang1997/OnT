from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall

logger = logging.getLogger(__name__)


class HierarchyTransformerLoss(torch.nn.Module):
    """Hyperbolic loss that linearly combines hyperbolic clustering loss and hyperbolic centripetal loss."""

    def __init__(
        self,
        model,
        clustering_loss_weight: float = 1.0,
        clustering_loss_margin: float = 5.0,
        centripetal_loss_weight: float = 1.0,
        centripetal_loss_margin: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.manifold = self.model.manifold
        self.cluster_loss = HyperbolicClusteringLoss(self.model.manifold, clustering_loss_margin)
        self.centri_loss = HyperbolicCentripetalLoss(self.model.manifold, centripetal_loss_margin)
        self.cluster_weight = clustering_loss_weight
        self.centri_weight = centripetal_loss_weight

    def get_config_dict(self):
        config = {"distance_metric": f"PoincareBall(c={self.manifold.c}).dist and dist0"}
        config[HyperbolicClusteringLoss.__name__] = {
            "weight": self.cluster_weight,
            **self.cluster_loss.get_config_dict(),
        }
        config[HyperbolicCentripetalLoss.__name__] = {
            "weight": self.centri_weight,
            **self.centri_loss.get_config_dict(),
        }
        return config

    def forward(self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor):
        """Forward propagation following sentence_transformers.losses interface."""
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        assert len(reps) == 3
        rep_anchor, rep_positive, rep_negative = reps

        cluster_loss = self.cluster_loss(rep_anchor, rep_positive, rep_negative)
        centri_loss = self.centri_loss(rep_anchor, rep_positive, rep_negative)
        combined_loss = self.cluster_weight * cluster_loss + self.centri_weight * centri_loss

        return {
            "loss": combined_loss,
            "cluster_loss": cluster_loss,
            "centri_loss": centri_loss,
        }

    def from_tensor(self, rep_anchor, rep_positive, rep_negative):
        """Compute loss directly from embedding tensors."""
        cluster_loss = self.cluster_loss(rep_anchor, rep_positive, rep_negative)
        centri_loss = self.centri_loss(rep_anchor, rep_positive, rep_negative)
        combined_loss = self.cluster_weight * cluster_loss + self.centri_weight * centri_loss
        return {
            "loss": combined_loss,
            "cluster_loss": cluster_loss,
            "centri_loss": centri_loss,
        }


class HyperbolicClusteringLoss(torch.nn.Module):
    r"""Hyperbolic loss: d(child, parent) < d(child, negative)."""

    def __init__(self, manifold: PoincareBall, margin: float):
        super().__init__()
        self.manifold = manifold
        self.margin = margin

    def get_config_dict(self):
        return {
            "distance_metric": f"PoincareBall(c={self.manifold.c}).dist",
            "margin": self.margin,
        }

    def forward(self, rep_anchor: torch.Tensor, rep_positive: torch.Tensor, rep_negative: torch.Tensor):
        distances_positive = self.manifold.dist(rep_anchor, rep_positive)
        distances_negative = self.manifold.dist(rep_anchor, rep_negative)
        cluster_triplet_loss = F.relu(distances_positive - distances_negative + self.margin)
        return cluster_triplet_loss.mean()


class HyperbolicCentripetalLoss(torch.nn.Module):
    r"""Hyperbolic loss: d(child, origin) > d(parent, origin)."""

    def __init__(self, manifold: PoincareBall, margin: float):
        super().__init__()
        self.manifold = manifold
        self.margin = margin

    def get_config_dict(self):
        return {
            "distance_metric": f"PoincareBall(c={self.manifold.c}).dist0",
            "margin": self.margin,
        }

    def forward(self, rep_anchor: torch.Tensor, rep_positive: torch.Tensor, rep_negative: torch.Tensor):
        rep_anchor_hyper_norms = self.manifold.dist0(rep_anchor)
        rep_positive_hyper_norms = self.manifold.dist0(rep_positive)
        centri_triplet_loss = F.relu(self.margin + rep_positive_hyper_norms - rep_anchor_hyper_norms)
        return centri_triplet_loss.mean()
