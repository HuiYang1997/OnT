"""Tests for OntologyTransformer model: init, forward, encode, encode_existence, save/load."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from ont.hit import HierarchyTransformer
from ont.model import OntologyTransformer


@pytest.fixture
def model():
    """Create a small OntologyTransformer for testing."""
    base = HierarchyTransformer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return OntologyTransformer(base, role_emd_mode="sentenceEmbedding", role_model_mode="rotation")


class TestOntologyTransformerInit:
    def test_init_rotation(self, model):
        assert model.role_model_mode == "rotation"
        assert model.dim % 2 == 0
        # role_model output: 1 (scaling) + dim//2 (rotation angles)
        assert model.role_model.out_features == 1 + model.dim // 2

    def test_init_transition(self):
        base = HierarchyTransformer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        m = OntologyTransformer(base, role_model_mode="transition")
        assert m.role_model.out_features == 1 + m.dim

    def test_properties(self, model):
        assert model.embed_dim > 0
        assert model.manifold is not None
        assert model.get_sentence_embedding_dimension() > 0


class TestEncode:
    def test_encode_single(self, model):
        emb = model.encode("hello world")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (model.dim,)

    def test_encode_batch(self, model):
        emb = model.encode(["hello", "world", "test"])
        assert emb.shape == (3, model.dim)

    def test_encode_tensor(self, model):
        emb = model.encode("hello", convert_to_tensor=True)
        assert isinstance(emb, torch.Tensor)


class TestEncodeExistence:
    def test_encode_existence_single(self, model):
        emb = model.encode_existence("eats", "food")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (1, model.dim)

    def test_encode_existence_batch(self, model):
        emb = model.encode_existence(["eats", "has part"], ["food", "cell"])
        assert emb.shape == (2, model.dim)

    def test_encode_existence_differs_from_concept(self, model):
        """∃r.C embedding should differ from plain concept C embedding."""
        concept_emb = model.encode("food")
        exist_emb = model.encode_existence("eats", "food")
        assert not np.allclose(concept_emb, exist_emb[0], atol=1e-3)


class TestSaveLoad:
    def test_save_load_roundtrip(self, model):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            model.save(save_path)

            # Check files exist
            assert os.path.exists(os.path.join(save_path, "wrapper_config.json"))
            assert os.path.exists(os.path.join(save_path, "role_model.pt"))

            # Load and compare
            loaded = OntologyTransformer.from_pretrained(save_path)
            assert loaded.role_emd_mode == model.role_emd_mode
            assert loaded.role_model_mode == model.role_model_mode

            # Embeddings should match
            emb1 = model.encode("test sentence")
            emb2 = loaded.encode("test sentence")
            np.testing.assert_allclose(emb1, emb2, atol=1e-5)

            # Existence embeddings should match
            exist1 = model.encode_existence("eats", "food")
            exist2 = loaded.encode_existence("eats", "food")
            np.testing.assert_allclose(exist1, exist2, atol=1e-5)
