"""Regression tests for Sentence Transformers inference API compatibility."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ont.model import OntologyTransformer


class StubHierarchyTransformer(nn.Module):
    """Small local stand-in that reproduces Sentence Transformers 5 inference output."""

    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @torch.inference_mode()
    def encode(self, sentences, **kwargs):
        size = 1 if isinstance(sentences, str) else len(sentences)
        embeddings = torch.ones(size, self.embed_dim)
        return embeddings[0] if isinstance(sentences, str) else embeddings

    def tokenize(self, sentences):
        return {
            "sentence_embedding": torch.ones(len(sentences), self.embed_dim),
            "prompt_length": 1,
        }

    def forward(self, features):
        return {"sentence_embedding": features["sentence_embedding"]}


def test_encode_roles_accepts_inference_tensors():
    model = OntologyTransformer(StubHierarchyTransformer())

    rotation, scaling = model.encode_roles("has part")

    assert rotation.shape == (model.dim // 2,)
    assert scaling.shape == ()
    assert not rotation.requires_grad
    assert not scaling.requires_grad


def test_encode_existence_preserves_non_tensor_tokenizer_metadata():
    model = OntologyTransformer(StubHierarchyTransformer())

    embedding = model.encode_existence("has part", "cell")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, model.dim)
