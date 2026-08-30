"""Integration test with SNOMED CT ontology."""
from __future__ import annotations

import os

import numpy as np
import pytest

from tests.conftest import SNOMEDCT_PATH


@pytest.mark.integration
@pytest.mark.timeout(1200)
@pytest.mark.skipif(not os.path.exists(SNOMEDCT_PATH), reason=f"SNOMED CT ontology not found at {SNOMEDCT_PATH}")
class TestSnomedCT:
    def test_prepare_snomedct(self, tmp_dir):
        """Test data preparation from snomedct.ofn."""
        from ont.data.prepare import prepare_ontology_data
        import json

        out_dir = os.path.join(tmp_dir, "snomedct_data")
        prepare_ontology_data(SNOMEDCT_PATH, out_dir)

        assert os.path.exists(os.path.join(out_dir, "train.jsonl"))
        assert os.path.exists(os.path.join(out_dir, "concept_names.json"))

        with open(os.path.join(out_dir, "concept_names.json")) as f:
            cn = json.load(f)
        assert len(cn) > 1000, f"Expected many concepts, got {len(cn)}"

    def test_fit_snomedct_short(self, tmp_dir):
        """Short training run on SNOMED CT (1 epoch, small batch)."""
        from ont.model import OntologyTransformer

        out_dir = os.path.join(tmp_dir, "snomedct_out")
        model = OntologyTransformer.fit(
            owl_path=SNOMEDCT_PATH,
            output_dir=out_dir,
            num_epochs=1,
            batch_size=16,
            eval_batch_size=16,
            base_model="sentence-transformers/all-MiniLM-L6-v2",
        )

        assert model is not None
        emb = model.encode("clinical finding")
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] > 0
