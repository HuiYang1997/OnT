"""Integration test with FoodOn ontology."""
from __future__ import annotations

import os

import numpy as np
import pytest

from tests.conftest import FOODON_PATH


@pytest.mark.integration
@pytest.mark.timeout(600)
@pytest.mark.skipif(not os.path.exists(FOODON_PATH), reason=f"FoodOn ontology not found at {FOODON_PATH}")
class TestFoodOn:
    def test_prepare_foodon(self, tmp_dir):
        """Test data preparation from foodon.ofn."""
        from ont.data.prepare import prepare_ontology_data
        import json

        out_dir = os.path.join(tmp_dir, "foodon_data")
        prepare_ontology_data(FOODON_PATH, out_dir)

        # Check files exist
        assert os.path.exists(os.path.join(out_dir, "train.jsonl"))
        assert os.path.exists(os.path.join(out_dir, "concept_names.json"))

        with open(os.path.join(out_dir, "concept_names.json")) as f:
            cn = json.load(f)
        # FoodOn should have thousands of concepts
        assert len(cn) > 100, f"Expected many concepts, got {len(cn)}"

    def test_fit_foodon_short(self, tmp_dir):
        """Short training run on FoodOn (1 epoch, small batch)."""
        from ont.model import OntologyTransformer

        out_dir = os.path.join(tmp_dir, "foodon_out")
        model = OntologyTransformer.fit(
            owl_path=FOODON_PATH,
            output_dir=out_dir,
            num_epochs=1,
            batch_size=16,
            eval_batch_size=16,
            base_model="sentence-transformers/all-MiniLM-L6-v2",
        )

        assert model is not None
        emb = model.encode("food product")
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] > 0

        exist_emb = model.encode_existence("has ingredient", "sugar")
        assert exist_emb.shape[1] == model.dim
