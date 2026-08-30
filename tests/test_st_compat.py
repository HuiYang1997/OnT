"""Offline regression tests for SentenceTransformers compatibility."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, models
from transformers import BertConfig, BertModel, BertTokenizerFast

from ont.hit import HierarchyTransformer
from ont.model import OntologyTransformer


def _save_tiny_sentence_transformer(tmp_path: Path) -> Path:
    """Build and save a one-layer, 32-dimensional BERT with mean pooling."""
    transformer_path = tmp_path / "bert"
    transformer_path.mkdir()

    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "has",
        "part",
        "cell",
        "membrane",
    ]
    (transformer_path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    BertModel(config).save_pretrained(transformer_path)
    BertTokenizerFast(
        vocab_file=str(transformer_path / "vocab.txt"),
        do_lower_case=True,
    ).save_pretrained(transformer_path)

    transformer = models.Transformer(str(transformer_path), max_seq_length=32)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode="mean",
    )
    model_path = tmp_path / "sentence-transformer"
    SentenceTransformer(modules=[transformer, pooling]).save(str(model_path))
    return model_path


def test_sentence_transformers_compatibility(tmp_path: Path):
    model_path = _save_tiny_sentence_transformer(tmp_path)
    base_model = HierarchyTransformer.from_pretrained(str(model_path))
    model = OntologyTransformer(base_model)

    rotation, scaling = model.encode_roles(["has part"])
    existence = model.encode_existence(["has part"], ["cell membrane"])

    assert rotation.shape == (1, 16)
    assert scaling.shape == (1,)
    assert existence.shape == (1, 32)
    assert np.isfinite(existence).all()
