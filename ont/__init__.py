"""Ontology-Transformer: End-to-end ontology embedding with hyperbolic geometry and role-based rotation."""
from __future__ import annotations

from ont.model import OntologyTransformer
from ont.pipeline import fit

__version__ = "0.1.7"

__all__ = [
    "OntologyTransformer",
    "fit",
]
