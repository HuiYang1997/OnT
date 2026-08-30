"""Tests for end-to-end pipeline on tiny synthetic ontology."""
from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.mark.timeout(300)
class TestPipeline:
    def test_fit_tiny_owl(self, tiny_owl_path, tmp_dir):
        """End-to-end: tiny OWL → prepare → train (1 epoch) → encode."""
        from ont.model import OntologyTransformer

        out_dir = os.path.join(tmp_dir, "pipeline_out")
        model = OntologyTransformer.fit(
            owl_path=tiny_owl_path,
            output_dir=out_dir,
            num_epochs=1,
            batch_size=4,
            eval_batch_size=4,
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            balanced=False,
        )

        assert model is not None

        # best_lambda should be set (determined from eval)
        assert model.best_lambda is not None
        assert isinstance(model.best_lambda, float)

        # Should be able to encode
        emb = model.encode("cat")
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] > 0

        # Should be able to encode existence
        exist_emb = model.encode_existence("eats", "food")
        assert exist_emb.shape == (1, model.dim)

        # Final model should be saved
        final_dir = os.path.join(out_dir, "final")
        assert os.path.exists(final_dir)

        # Should be able to load and best_lambda should persist
        loaded = OntologyTransformer.from_pretrained(final_dir)
        assert loaded.best_lambda == model.best_lambda
        emb2 = loaded.encode("cat")
        np.testing.assert_allclose(emb, emb2, atol=1e-5)

    def test_fit_balanced(self, tiny_owl_path, tmp_dir):
        """End-to-end with balanced=True."""
        from ont.model import OntologyTransformer

        out_dir = os.path.join(tmp_dir, "balanced_out")
        model = OntologyTransformer.fit(
            owl_path=tiny_owl_path,
            output_dir=out_dir,
            num_epochs=1,
            batch_size=4,
            eval_batch_size=4,
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            balanced=True,
            balanced_negatives=1,
        )
        assert model is not None
        assert model.best_lambda is not None
        emb = model.encode("dog")
        assert emb.shape[0] > 0
