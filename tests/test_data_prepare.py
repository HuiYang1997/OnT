"""Tests for data preparation: OWL → OnT training files using tiny synthetic ontology."""
from __future__ import annotations

import json
import os

import pytest


@pytest.mark.timeout(120)
class TestPrepareOntologyData:
    def test_prepare_tiny_owl(self, tiny_owl_path, tmp_dir):
        """Test that prepare_ontology_data produces expected output files from tiny OWL."""
        from ont.data.prepare import prepare_ontology_data

        out_dir = os.path.join(tmp_dir, "ont_data")
        prepare_ontology_data(tiny_owl_path, out_dir)

        # Check required files exist (no test.json by design)
        required_files = [
            "train.jsonl",
            "train_exist.jsonl",
            "train_conj.jsonl",
            "val.json",
            "concept_names.json",
            "role_names.json",
            "role_inverse.json",
        ]
        for fname in required_files:
            fpath = os.path.join(out_dir, fname)
            assert os.path.exists(fpath), f"Missing file: {fname}"

        # test.json should NOT exist (no test split by default)
        assert not os.path.exists(os.path.join(out_dir, "test.json"))

        # Check concept_names.json is valid
        with open(os.path.join(out_dir, "concept_names.json")) as f:
            cn = json.load(f)
        assert len(cn) >= 5, f"Expected at least 5 concepts, got {len(cn)}"

        # Check role_names.json is valid
        with open(os.path.join(out_dir, "role_names.json")) as f:
            rn = json.load(f)
        assert len(rn) >= 2, f"Expected at least 2 roles, got {len(rn)}"

        # Check train.jsonl has entries — all axioms go to train
        with open(os.path.join(out_dir, "train.jsonl")) as f:
            train_lines = [l.strip() for l in f if l.strip()]
        assert len(train_lines) > 0, "train.jsonl is empty"

        for line in train_lines:
            entry = json.loads(line)
            assert "child" in entry
            assert "parent" in entry
            assert "negative" in entry

        # Check train_exist.jsonl
        with open(os.path.join(out_dir, "train_exist.jsonl")) as f:
            exist_lines = [l.strip() for l in f if l.strip()]
        for line in exist_lines:
            entry = json.loads(line)
            assert "Concept" in entry
            assert "role" in entry
            assert "con" in entry

        # Check val.json structure (sampled from train)
        with open(os.path.join(out_dir, "val.json")) as f:
            val_data = json.load(f)
        assert "query_sentences" in val_data
        assert "answer_ids" in val_data
        for k in ["nf1", "nf2", "nf3", "nf4"]:
            assert k in val_data["query_sentences"]
            assert k in val_data["answer_ids"]

        # Val should have some entries (10% of all axioms)
        total_val = sum(len(v) for v in val_data["query_sentences"].values())
        assert total_val > 0, "val.json has no eval queries"

    def test_no_axiom_duplication(self, tiny_owl_path, tmp_dir):
        """Axiom counts must match expected values — no duplication from getImportsClosure.

        The tiny ontology has exactly:
            - 3 nf1 (SubClassOf with atomic classes): Cat⊑Animal, Dog⊑Animal, Cat⊑Pet
            - 0 nf2 (conjunction on LHS)
            - 2 nf3 (existential on RHS): Cat⊑∃eats.Food, Dog⊑∃eats.Food
            - 0 nf4 (existential on LHS)
        Total training samples = 5.
        """
        from ont.data.prepare import ELNormalizedData

        el = ELNormalizedData()
        el.output_dir = os.path.join(tmp_dir, "axiom_count_check")
        os.makedirs(el.output_dir, exist_ok=True)

        ont, verbalizer = el.load(tiny_owl_path)
        el.create_dataset(ont, verbalizer)

        assert len(el.nf1_org) == 3, f"Expected 3 nf1 axioms, got {len(el.nf1_org)}"
        assert len(el.nf2_org) == 0, f"Expected 0 nf2 axioms, got {len(el.nf2_org)}"
        assert len(el.nf3_org) == 2, f"Expected 2 nf3 axioms, got {len(el.nf3_org)}"
        assert len(el.nf4_org) == 0, f"Expected 0 nf4 axioms, got {len(el.nf4_org)}"

        # Also verify train.jsonl line count matches
        el.save_ont_data(eval_ratio=0.0, max_eval=0)
        with open(os.path.join(el.output_dir, "train.jsonl")) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 5, f"Expected 5 train lines, got {len(lines)}"

    def test_prepare_idempotent(self, tiny_owl_path, tmp_dir):
        """Running prepare twice should produce consistent results."""
        from ont.data.prepare import prepare_ontology_data

        out1 = os.path.join(tmp_dir, "run1")
        out2 = os.path.join(tmp_dir, "run2")
        prepare_ontology_data(tiny_owl_path, out1)
        prepare_ontology_data(tiny_owl_path, out2)

        with open(os.path.join(out1, "concept_names.json")) as f:
            cn1 = json.load(f)
        with open(os.path.join(out2, "concept_names.json")) as f:
            cn2 = json.load(f)
        assert cn1 == cn2


class TestLoadLocalDataset:
    def test_load_prepared_data(self, tiny_owl_path, tmp_dir):
        """Test loading data prepared from tiny OWL."""
        from ont.data.prepare import prepare_ontology_data
        from ont.data.load import load_local_dataset

        out_dir = os.path.join(tmp_dir, "ont_data")
        prepare_ontology_data(tiny_owl_path, out_dir)

        dataset = load_local_dataset(out_dir)

        assert "train" in dataset
        assert "train_exist" in dataset
        assert "train_conj" in dataset
        assert "val" in dataset
        assert "test" in dataset  # empty structure, not a file
        assert "concept_names" in dataset

        assert len(dataset["train"]) > 0
        assert "name" in dataset["concept_names"].column_names

        # val should have query_sentences/answer_ids
        assert "query_sentences" in dataset["val"]
        assert "answer_ids" in dataset["val"]

        # test should be empty (no test.json generated)
        total_test = sum(len(v) for v in dataset["test"]["query_sentences"].values())
        assert total_test == 0


@pytest.mark.timeout(120)
class TestAxiomCounts:
    """Tests that axiom counts are correct — catches duplication bugs."""

    def test_comprehensive_all_normal_forms(self, comprehensive_owl_path, tmp_dir):
        """Verify exact nf1-nf4 counts for the comprehensive ontology.

        Expected (see conftest.py comprehensive_owl_path for derivation):
            nf1=5, nf2=2, nf3=2, nf4=1
        """
        from ont.data.prepare import ELNormalizedData

        el = ELNormalizedData()
        el.output_dir = os.path.join(tmp_dir, "comp_data")
        os.makedirs(el.output_dir, exist_ok=True)

        ont, verbalizer = el.load(comprehensive_owl_path)
        el.create_dataset(ont, verbalizer)

        assert len(el.nf1_org) == 5, f"Expected 5 nf1, got {len(el.nf1_org)}"
        assert len(el.nf2_org) == 2, f"Expected 2 nf2, got {len(el.nf2_org)}"
        assert len(el.nf3_org) == 2, f"Expected 2 nf3, got {len(el.nf3_org)}"
        assert len(el.nf4_org) == 1, f"Expected 1 nf4, got {len(el.nf4_org)}"

    def test_train_count_matches_axiom_count(self, comprehensive_owl_path, tmp_dir):
        """Total train.jsonl lines must equal sum of nf1+nf2+nf3+nf4."""
        from ont.data.prepare import prepare_ontology_data, ELNormalizedData

        out_dir = os.path.join(tmp_dir, "comp_train")
        prepare_ontology_data(comprehensive_owl_path, out_dir)

        with open(os.path.join(out_dir, "train.jsonl")) as f:
            train_lines = [l for l in f if l.strip()]

        # Total axioms = nf1 + nf2 + nf3 + nf4 = 5 + 2 + 2 + 1 = 10
        assert len(train_lines) == 10, (
            f"Expected 10 train lines (5+2+2+1), got {len(train_lines)}"
        )
