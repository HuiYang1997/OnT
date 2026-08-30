from __future__ import annotations

import torch
from geoopt.manifolds import PoincareBall


def get_circum_poincareball(embed_dim: int) -> PoincareBall:
    """Get a Poincaré Ball with a curvature adapted to a given embedding dimension so that it circumscribes the output embedding space of pre-trained language models."""
    curvature = 1 / embed_dim
    manifold = PoincareBall(c=curvature)
    return manifold
