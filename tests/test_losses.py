"""Tests for HiT loss and LogicalConstraintLoss (balanced & non-balanced)."""
from __future__ import annotations

import pytest
import torch
from geoopt.manifolds import PoincareBall

from ont.hit import HierarchyTransformer
from ont.losses.hit_loss import HierarchyTransformerLoss, HyperbolicClusteringLoss, HyperbolicCentripetalLoss
from ont.model import OntologyTransformer


@pytest.fixture
def manifold():
    return PoincareBall(c=1 / 384)


@pytest.fixture
def model():
    base = HierarchyTransformer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return OntologyTransformer(base)


class TestHyperbolicClusteringLoss:
    def test_output_shape(self, manifold):
        loss_fn = HyperbolicClusteringLoss(manifold, margin=3.0)
        anchor = torch.randn(4, 384) * 0.1
        pos = torch.randn(4, 384) * 0.1
        neg = torch.randn(4, 384) * 0.1
        loss = loss_fn(anchor, pos, neg)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_case(self, manifold):
        """When positive is closer than negative, loss should be small."""
        loss_fn = HyperbolicClusteringLoss(manifold, margin=0.1)
        anchor = torch.zeros(4, 384)
        pos = torch.ones(4, 384) * 0.01
        neg = torch.ones(4, 384) * 0.5
        loss = loss_fn(anchor, pos, neg)
        assert loss.item() < 1.0


class TestHyperbolicCentripetalLoss:
    def test_output_shape(self, manifold):
        loss_fn = HyperbolicCentripetalLoss(manifold, margin=0.5)
        anchor = torch.randn(4, 384) * 0.1
        pos = torch.randn(4, 384) * 0.1
        neg = torch.randn(4, 384) * 0.1
        loss = loss_fn(anchor, pos, neg)
        assert loss.shape == ()
        assert loss.item() >= 0


class TestHierarchyTransformerLoss:
    def test_forward(self, model):
        hit_loss = HierarchyTransformerLoss(model=model.hit_model)
        sentences = ["cat", "animal", "car"]
        device = next(model.parameters()).device
        features = [
            model.hit_model.tokenize([s]) for s in sentences
        ]
        features = [
            {k: v.to(device) if hasattr(v, "to") else v for k, v in f.items()}
            for f in features
        ]
        result = hit_loss(features, labels=None)
        assert "loss" in result
        assert "cluster_loss" in result
        assert "centri_loss" in result
        assert result["loss"].shape == ()

    def test_from_tensor(self, model):
        hit_loss = HierarchyTransformerLoss(model=model.hit_model)
        anchor = torch.randn(4, model.dim) * 0.1
        pos = torch.randn(4, model.dim) * 0.1
        neg = torch.randn(4, model.dim) * 0.1
        result = hit_loss.from_tensor(anchor, pos, neg)
        assert "loss" in result

    def test_config_dict(self, model):
        hit_loss = HierarchyTransformerLoss(model=model.hit_model)
        cfg = hit_loss.get_config_dict()
        assert "distance_metric" in cfg
