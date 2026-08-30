from __future__ import annotations

import json
import os
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn

from ont.hit import HierarchyTransformer


class OntologyTransformer(nn.Module):
    """Ontology Transformer model that extends HierarchyTransformer with role-based rotation
    for existential restriction embeddings (∃r.C).

    Public API:
        - encode(sentences) → numpy array of embeddings
        - encode_existence(role_sentence, concept_sentence) → numpy array of ∃r.C embeddings
        - from_pretrained(path) → load a saved model
        - save(path) → save model to disk
        - fit(owl_path, ...) → end-to-end: OWL → data → train → model (class method)
    """

    def __init__(self, base_model: HierarchyTransformer, role_emd_mode: str = "sentenceEmbedding", role_model_mode: str = "rotation"):
        super().__init__()
        self.hit_model = base_model
        self.dim = self.hit_model.embed_dim
        self.role_emd_mode = role_emd_mode
        self.role_model_mode = role_model_mode
        self.best_lambda: float | None = None  # determined during eval, saved with model

        assert self.dim % 2 == 0, "Embedding dimension must be even"

        # Define role_model as a linear layer for role transformation
        if self.role_model_mode == "rotation":
            output_dim = self.dim // 2
        elif self.role_model_mode == "transition":
            output_dim = self.dim
        else:
            raise ValueError(f"Unknown role_model_mode: {self.role_model_mode}")

        self.role_model = nn.Linear(self.dim, 1 + output_dim)
        self.margin = 0.1

    # ------------------------------------------------------------------ #
    #  Public inference API
    # ------------------------------------------------------------------ #

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 64,
        convert_to_tensor: bool = False,
        **kwargs,
    ):
        """Encode sentences into embeddings.

        Args:
            sentences: A single sentence string or a list of sentence strings.
            batch_size: Batch size for encoding.
            convert_to_tensor: If True, return a torch.Tensor; otherwise numpy array.

        Returns:
            Embeddings as numpy array or torch.Tensor.
        """
        return self.hit_model.encode(
            sentences,
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor,
            **kwargs,
        )

    def encode_existence(
        self,
        role_sentences: Union[str, List[str]],
        concept_sentences: Union[str, List[str]],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode ∃r.C concepts via rotation: apply learned f_r on C's embedding.

        Args:
            role_sentences: Role sentence(s) for r.
            concept_sentences: Concept sentence(s) for C.
            batch_size: Batch size for encoding.

        Returns:
            Numpy array of rotated embeddings representing ∃r.C.
        """
        if isinstance(role_sentences, str):
            role_sentences = [role_sentences]
        if isinstance(concept_sentences, str):
            concept_sentences = [concept_sentences]

        assert len(role_sentences) == len(concept_sentences), \
            "role_sentences and concept_sentences must have the same length"

        device = next(self.hit_model.parameters()).device
        self.role_model = self.role_model.to(device)
        all_results = []

        for start in range(0, len(role_sentences), batch_size):
            end = min(start + batch_size, len(role_sentences))
            role_batch = role_sentences[start:end]
            con_batch = concept_sentences[start:end]

            role_features = self.hit_model.tokenize(role_batch)
            role_features = {k: v.to(device) if hasattr(v, "to") else v for k, v in role_features.items()}
            con_features = self.hit_model.tokenize(con_batch)
            con_features = {k: v.to(device) if hasattr(v, "to") else v for k, v in con_features.items()}

            with torch.no_grad():
                emb = self.existence_emb([role_features, con_features])
                all_results.append(emb.cpu().numpy())

        return np.concatenate(all_results, axis=0)

    # ------------------------------------------------------------------ #
    #  Encode helpers for roles (used by encode_existence and training)
    # ------------------------------------------------------------------ #

    def encode_roles(self, role_sentences: Union[str, List[str]]):
        """Encode role sentences into (rotation, scaling) via the role model."""
        with torch.no_grad():
            role_embed = self.role_model(
                self.hit_model.encode(role_sentences, convert_to_tensor=True, show_progress_bar=False)
            )
            if self.role_model_mode == "rotation":
                rotation = torch.tanh(role_embed[..., 1:]) * torch.pi
            else:
                rotation = role_embed[..., 1:]
            scaling = torch.exp(role_embed[..., 0])
            return rotation, scaling

    # ------------------------------------------------------------------ #
    #  Internal forward methods (used during training)
    # ------------------------------------------------------------------ #

    def forward(self, features) -> Dict[str, torch.Tensor]:
        """Forward pass that maintains compatibility with HierarchyTransformer interface."""
        return self.hit_model(features)

    def get_role_embedding(self, role_feature):
        """Compute role rotation and scaling from tokenized role features."""
        if self.role_emd_mode == "tokenEmbedding":
            token_embeddings = self.hit_model._first_module().auto_model.embeddings.word_embeddings(role_feature["input_ids"])
            attention_mask = role_feature["attention_mask"].unsqueeze(-1)
            role_emb = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            role_emb = self.role_model(role_emb)
        elif self.role_emd_mode == "sentenceEmbedding":
            role_emb = self.role_model(
                self.hit_model(role_feature)["sentence_embedding"]
            )
        else:
            raise ValueError(f"Unknown role_emd_mode: {self.role_emd_mode}")

        if self.role_model_mode == "rotation":
            rotation = torch.tanh(role_emb[..., 1:]) * torch.pi
        else:
            rotation = role_emb[..., 1:]

        scaling_emb = torch.exp(role_emb[..., 0])
        return scaling_emb, rotation

    def pair_loss_existence(self, features):
        """Pair-based existence loss (alternative to hit-based)."""
        complex_emd = self.hit_model(features[0])["sentence_embedding"]
        con_emd = self.hit_model(features[2])["sentence_embedding"]

        slide_dist_con = self.hit_model.manifold.dist(con_emd[:-1], con_emd[1:])
        slide_dist_complex = self.hit_model.manifold.dist(complex_emd[:-1], complex_emd[1:])
        loss_dist_hyp = torch.relu(slide_dist_complex - slide_dist_con + self.margin)

        complex_hyper_norms = self.hit_model.manifold.dist0(complex_emd)
        con_hyper_norms = self.hit_model.manifold.dist0(con_emd)
        slid_norm_con = con_hyper_norms[:-1] - con_hyper_norms[1:]
        slid_norm_complex = complex_hyper_norms[:-1] - complex_hyper_norms[1:]
        loss_dist_norm = torch.relu(slid_norm_complex - slid_norm_con + self.margin)

        return (loss_dist_hyp + loss_dist_norm).mean()

    def existence_emb(self, features) -> torch.Tensor:
        """Compute ∃r.C embedding via rotation from tokenized [role_features, concept_features]."""
        con_emd = self.hit_model(features[1])["sentence_embedding"]
        scaling, rotation = self.get_role_embedding(features[0])

        if self.role_model_mode == "rotation":
            con_emb1, con_emb2 = torch.chunk(con_emd, 2, dim=-1)
            sin_theta = torch.sin(rotation)
            cos_theta = torch.cos(rotation)
            rotated_emb1 = con_emb1 * cos_theta - con_emb2 * sin_theta
            rotated_emb2 = con_emb1 * sin_theta + con_emb2 * cos_theta
            rotated_emb = torch.cat([rotated_emb1, rotated_emb2], dim=-1)
        elif self.role_model_mode == "transition":
            rotated_emb = self.hit_model.manifold.expmap(con_emd, rotation)
        else:
            raise ValueError(f"Unknown role_model_mode: {self.role_model_mode}")

        # scaling (not used in final output but kept for future)
        # scaled_emb = self.hit_model.manifold.mobius_scalar_mul(scaling.unsqueeze(-1), rotated_emb)
        return rotated_emb

    def forward_existence(self, features) -> Dict[str, torch.Tensor]:
        """Output embedding of concept of the form ∃r.C by rotation and text description."""
        concept_emd = self.hit_model(features[0])["sentence_embedding"]
        scaled_emb = self.existence_emb(features[1:])
        return {"sentence_embedding": concept_emd, "rotated_embedding": scaled_emb}

    def score_hierarchy(self, child_embeds: torch.Tensor, parent_embeds: torch.Tensor, weight: float = 0.0) -> torch.Tensor:
        """Score hierarchical relationships between entities."""
        distances = self.manifold.dist(child_embeds, parent_embeds)
        child_norms = self.manifold.dist0(child_embeds)
        parent_norms = self.manifold.dist0(parent_embeds)
        dist_norm = distances + weight * (parent_norms - child_norms)
        return -dist_norm

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def manifold(self):
        return self.hit_model.manifold

    @property
    def embed_dim(self):
        return self.hit_model.embed_dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.hit_model.get_sentence_embedding_dimension()

    def tokenize(self, *args, **kwargs):
        return self.hit_model.tokenize(*args, **kwargs)

    def get_config_dict(self):
        return self.hit_model.get_config_dict()

    # ------------------------------------------------------------------ #
    #  Save / Load
    # ------------------------------------------------------------------ #

    def save(self, output_path: str, *args, **kwargs):
        """Save the model with all its attributes."""
        if output_path is None or not isinstance(output_path, str):
            raise ValueError("'output_path' must be provided as the first argument and must be a string")

        os.makedirs(output_path, exist_ok=True)
        self.hit_model.save(output_path, *args, **kwargs)

        wrapper_config = {
            "role_emd_mode": self.role_emd_mode,
            "role_model_mode": self.role_model_mode,
            "best_lambda": self.best_lambda,
        }
        with open(os.path.join(output_path, "wrapper_config.json"), "w") as f:
            json.dump(wrapper_config, f)

        torch.save(self.role_model.state_dict(), os.path.join(output_path, "role_model.pt"))
        return output_path

    @classmethod
    def from_pretrained(cls, input_path: str, revision: str | None = None):
        """Load a saved OntologyTransformer model."""
        base_model = HierarchyTransformer.from_pretrained(input_path, revision=revision)

        wrapper_config_path = os.path.join(input_path, "wrapper_config.json")
        if os.path.exists(wrapper_config_path):
            with open(wrapper_config_path, "r") as f:
                wrapper_config = json.load(f)
            role_emd_mode = wrapper_config.get("role_emd_mode", "sentenceEmbedding")
            role_model_mode = wrapper_config.get("role_model_mode", "rotation")
        else:
            role_emd_mode = "sentenceEmbedding"
            role_model_mode = "rotation"

        ont_model = cls(base_model, role_emd_mode, role_model_mode)
        ont_model.best_lambda = wrapper_config.get("best_lambda", None) if os.path.exists(wrapper_config_path) else None

        role_model_path = os.path.join(input_path, "role_model.pt")
        if os.path.exists(role_model_path):
            device = next(base_model.parameters()).device
            ont_model.role_model.load_state_dict(torch.load(role_model_path, map_location=device))

        device = next(base_model.parameters()).device
        ont_model.role_model = ont_model.role_model.to(device)
        return ont_model

    # ------------------------------------------------------------------ #
    #  End-to-end class method
    # ------------------------------------------------------------------ #

    @classmethod
    def fit(
        cls,
        owl_path: str,
        output_dir: str = "./ont_output",
        eval_owl_path: str | None = None,
        test_owl_path: str | None = None,
        balanced: bool = False,
        balanced_negatives: int = 1,
        num_epochs: int = 1,
        batch_size: int = 64,
        eval_batch_size: int = 32,
        learning_rate: float = 1e-5,
        base_model: str = "sentence-transformers/all-MiniLM-L12-v2",
        role_emd_mode: str = "sentenceEmbedding",
        role_model_mode: str = "rotation",
        existence_loss_kind: str = "hit",
        clustering_loss_weight: float = 1.0,
        clustering_loss_margin: float = 3.0,
        centripetal_loss_weight: float = 1.0,
        centripetal_loss_margin: float = 0.5,
        conj_weight: float = 1.0,
        exist_weight: float = 1.0,
        eval_ratio: float = 0.1,
        max_eval: int = 1000,
        seed: int = 8888,
    ) -> "OntologyTransformer":
        """End-to-end: OWL → prepare data → train → return model.

        All axioms are used for training. 10 % (max 1000) are sampled as
        evaluation data to determine best_lambda. No test unless
        *test_owl_path* is given.

        Args:
            owl_path: Path to OWL/OFN ontology file for training.
            output_dir: Directory for output data and model checkpoints.
            eval_owl_path: Optional separate OWL for evaluation.
            test_owl_path: Optional separate OWL for testing.
            balanced: Use balanced training with extra C_neg contrastive loss.
            balanced_negatives: Number of negative concept samples for balanced mode.
            num_epochs: Number of training epochs.
            batch_size: Training batch size.
            eval_batch_size: Evaluation batch size.
            learning_rate: Learning rate.
            base_model: Pretrained SentenceBERT model name or path.
            role_emd_mode: Role embedding mode.
            role_model_mode: Role model mode.
            existence_loss_kind: Existence loss type.
            clustering_loss_weight: Weight for clustering loss.
            clustering_loss_margin: Margin for clustering loss.
            centripetal_loss_weight: Weight for centripetal loss.
            centripetal_loss_margin: Margin for centripetal loss.
            conj_weight: Weight for conjunction loss.
            exist_weight: Weight for existence loss.
            eval_ratio: Fraction of axioms to sample for evaluation.
            max_eval: Maximum number of evaluation samples.
            seed: Random seed.

        Returns:
            Trained OntologyTransformer model with best_lambda set.
        """
        from ont.pipeline import fit as _fit
        return _fit(
            owl_path=owl_path,
            output_dir=output_dir,
            eval_owl_path=eval_owl_path,
            test_owl_path=test_owl_path,
            balanced=balanced,
            balanced_negatives=balanced_negatives,
            num_epochs=num_epochs,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            base_model=base_model,
            role_emd_mode=role_emd_mode,
            role_model_mode=role_model_mode,
            existence_loss_kind=existence_loss_kind,
            clustering_loss_weight=clustering_loss_weight,
            clustering_loss_margin=clustering_loss_margin,
            centripetal_loss_weight=centripetal_loss_weight,
            centripetal_loss_margin=centripetal_loss_margin,
            conj_weight=conj_weight,
            exist_weight=exist_weight,
            eval_ratio=eval_ratio,
            max_eval=max_eval,
            seed=seed,
        )
