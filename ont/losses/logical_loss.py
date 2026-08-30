from __future__ import annotations

import random
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogicalConstraintLoss(nn.Module):
    """Loss function that enforces logical constraints for ∃r.C and C⊓D concepts.

    Supports balanced training mode: when balanced=True, extra contrastive
    cluster loss is added to the subsumption loss using randomly sampled
    negative concepts C_neg.  Other losses (existential, conjunction) are
    not affected.
    """

    def __init__(
        self,
        model,
        hit_loss,
        data_conj,
        data_exist,
        batch_size: int,
        conj_weight: float = 1.0,
        exist_weight: float = 1.0,
        existence_loss_kind: str = "pair",
        balanced: bool = False,
        balanced_negatives: int = 1,
        all_concepts: list | None = None,
    ):
        super().__init__()
        self.model = model
        self.hit_loss = hit_loss
        self.manifold = self.model.hit_model.manifold

        self.conj_weight = conj_weight
        self.exist_weight = exist_weight
        self.batch_size = batch_size

        self.data_conj = data_conj
        self.data_exist = data_exist

        self.role_prompt = ""
        self.existence_loss_kind = existence_loss_kind

        self.balanced = balanced
        self.balanced_negatives = balanced_negatives
        self._all_concepts = all_concepts or []

        self.device = next(self.model.parameters()).device

    def select_conj(self):
        """Sample a batch of conjunction data and tokenize."""
        batch_indices = random.sample(range(len(self.data_conj)), min(self.batch_size, len(self.data_conj)))
        batch = self.data_conj.select(batch_indices)

        concepts = [item["Concept"] for item in batch]
        con1s = [item["con1"] for item in batch]
        con2s = [item["con2"] for item in batch]

        return [
            self.model.hit_model.tokenizer(concepts, return_tensors="pt", padding=True, truncation=True).to(self.device),
            self.model.hit_model.tokenizer(con1s, return_tensors="pt", padding=True, truncation=True).to(self.device),
            self.model.hit_model.tokenizer(con2s, return_tensors="pt", padding=True, truncation=True).to(self.device),
        ]

    def select_exist(self):
        """Sample a batch of existential data and tokenize."""
        batch_indices = random.sample(range(len(self.data_exist)), min(self.batch_size, len(self.data_exist)))
        batch = self.data_exist.select(batch_indices)

        concepts = [item["Concept"] for item in batch]
        roles = [self.role_prompt + item["role"] for item in batch]
        cons = [item["con"] for item in batch]

        return [
            self.model.hit_model.tokenizer(concepts, return_tensors="pt", padding=True, truncation=True).to(self.device),
            self.model.hit_model.tokenizer(roles, return_tensors="pt", padding=True, truncation=True).to(self.device),
            self.model.hit_model.tokenizer(cons, return_tensors="pt", padding=True, truncation=True).to(self.device),
        ]

    def select_negative_concepts(self, n: int):
        """Sample n random concept sentences from data_exist for balanced negative sampling."""
        all_cons = list({item["con"] for item in self.data_exist})
        if len(all_cons) < n:
            sampled = all_cons
        else:
            sampled = random.sample(all_cons, n)
        return sampled

    def _sample_negative_concepts(self, n: int):
        """Sample n random concept names for balanced subsumption loss."""
        pool = self._all_concepts
        if not pool:
            return []
        if len(pool) <= n:
            return list(pool)
        return random.sample(pool, n)

    def conj_loss(self, sentence_conjs, neg_samples):
        """Compute losses for conjunctions."""
        reps_conj = [self.model(sc)["sentence_embedding"] for sc in sentence_conjs]
        conj_emb, conj1_emb, conj2_emb = reps_conj

        # Truncate neg_samples to match conj_emb batch size (may be smaller when
        # the total conjunction dataset is smaller than batch_size)
        n = conj_emb.size(0)
        neg_samples_1 = neg_samples[torch.randperm(neg_samples.size(0))[:n]]
        neg_samples_2 = neg_samples[torch.randperm(neg_samples.size(0))[:n]]

        conj_loss = (
            self.hit_loss.from_tensor(conj_emb, conj1_emb, neg_samples_1)["loss"]
            + self.hit_loss.from_tensor(conj_emb, conj2_emb, neg_samples_2)["loss"]
        ) / 2

        return conj_loss

    def exist_loss(self, sentence_exists, neg_samples):
        """Compute losses for existential restrictions."""
        if self.existence_loss_kind == "hit":
            exist_emb = self.model.forward_existence(sentence_exists)
            sentence_emb = exist_emb["sentence_embedding"]
            composed_emb = exist_emb["rotated_embedding"]

            # Truncate neg_samples to match sentence_emb batch size
            n = sentence_emb.size(0)
            neg_samples_1 = neg_samples[torch.randperm(neg_samples.size(0))[:n]]
            neg_samples_2 = neg_samples[torch.randperm(neg_samples.size(0))[:n]]

            exist_loss = (
                self.hit_loss.from_tensor(sentence_emb, composed_emb, neg_samples_1)["loss"]
                + self.hit_loss.from_tensor(composed_emb, sentence_emb, neg_samples_2)["loss"]
            ) / 2
        elif self.existence_loss_kind == "pair":
            exist_loss = self.model.pair_loss_existence(sentence_exists)
        elif self.existence_loss_kind == "dist":
            exist_emb = self.model.forward_existence(sentence_exists)
            exist_sentence_emb = exist_emb["sentence_embedding"]
            exist_rotate_emb = exist_emb["rotated_embedding"]
            exist_loss = self.manifold.dist(exist_sentence_emb, exist_rotate_emb).mean()
        else:
            raise ValueError(f"Unknown existence_loss_kind: {self.existence_loss_kind}")

        return exist_loss

    def balanced_cluster_loss(self, sentence_features):
        """Extra subsumption cluster loss with sampled C_neg.

        Sample random concepts C_neg, encode them, and add a contrastive
        margin loss:  d(anchor, positive) < d(anchor, C_neg) + margin.
        This strengthens the subsumption embedding without affecting
        existential or conjunction losses.
        """
        anchor_emb = self.model(sentence_features[0])["sentence_embedding"]
        positive_emb = self.model(sentence_features[1])["sentence_embedding"]
        n = anchor_emb.size(0)

        balanced_loss = torch.tensor(0.0, device=self.device)
        for _ in range(self.balanced_negatives):
            neg_concepts = self._sample_negative_concepts(n)
            if not neg_concepts:
                continue
            neg_features = self.model.hit_model.tokenizer(
                neg_concepts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            neg_emb = self.model(neg_features)["sentence_embedding"]
            m = min(n, neg_emb.size(0))
            balanced_loss = balanced_loss + self.hit_loss.from_tensor(
                anchor_emb[:m], positive_emb[:m], neg_emb[:m]
            )["loss"]

        balanced_loss = balanced_loss / max(self.balanced_negatives, 1)
        return balanced_loss

    def forward(
        self,
        sentence_features,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the logical constraint loss."""
        neg_samples = self.model(sentence_features[2])["sentence_embedding"]
        self.batch_size = neg_samples.size(0)

        if len(self.data_exist) > 0:
            sentence_exists = self.select_exist()
            exist_loss = self.exist_loss(sentence_exists, neg_samples)
        else:
            exist_loss = torch.tensor(0.0, device=self.device)

        if self.data_conj and len(self.data_conj) > 0:
            sentence_conjs = self.select_conj()
            conj_loss = self.conj_loss(sentence_conjs, neg_samples)
        else:
            conj_loss = torch.tensor(0.0, device=self.device)

        hit_loss = self.hit_loss(sentence_features, labels)

        total_loss = self.conj_weight * conj_loss + self.exist_weight * exist_loss + hit_loss["loss"]

        if self.balanced:
            total_loss = total_loss + self.balanced_cluster_loss(sentence_features)

        return {
            "loss": total_loss,
            "conj_loss": conj_loss,
            "exist_loss": exist_loss,
            "cluster_loss": hit_loss["cluster_loss"],
            "centri_loss": hit_loss["centri_loss"],
        }

    def get_config_dict(self):
        return {
            "conj_weight": self.conj_weight,
            "exist_weight": self.exist_weight,
            "cluster_weight": self.hit_loss.cluster_weight,
            "centri_weight": self.hit_loss.centri_weight,
            "balanced": self.balanced,
            "balanced_negatives": self.balanced_negatives,
        }
