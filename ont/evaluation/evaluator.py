from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers.evaluation import SentenceEvaluator
from tqdm import tqdm

from ont.evaluation.ranking import RankingResult, dists_to_ranks, combine_rankings, compute_metrics

logger = logging.getLogger(__name__)


class OnTEvaluator(SentenceEvaluator):
    """Evaluator for OnT models on hierarchy prediction tasks."""

    # Default candidate chunk size: controls GPU memory per inference step.
    # 4096 × eval_batch × dim × 4 bytes ≈ manageable on any GPU ≥ 8 GB.
    DEFAULT_CAND_CHUNK = 4096

    def __init__(
        self,
        ont_model,
        query_entities: dict[str, list],
        answer_ids: dict[str, list[int]],
        all_entities: list[str],
        batch_size: int,
        cand_chunk_size: int = DEFAULT_CAND_CHUNK,
    ):
        super().__init__()
        self.primary_metric = "H@1"
        self.query_entities = query_entities
        self.answer_ids = answer_ids
        self.all_entities = all_entities
        self.batch_size = batch_size
        self.cand_chunk_size = cand_chunk_size
        self.results = pd.DataFrame(
            columns=["axiom_kind", "centri_weight", "H@1", "H@10", "H@100", "MRR", "MR", "median", "AUC"]
        )
        self.ont_model = ont_model
        self.inference_mode = "sentence"

    @torch.no_grad()
    def inference(self, model, centri_weight, child_embeds=None, parent_embeds=None):
        """Score hierarchical relationships."""
        dists = model.manifold.dist(child_embeds, parent_embeds)
        child_norms = model.manifold.dist0(child_embeds)
        parent_norms = model.manifold.dist0(parent_embeds)
        return -(dists + centri_weight * (parent_norms - child_norms))

    def calculate_metrics(self, all_candidates_cpu, model, centri_weight, kind="nf1", show_progress=False):
        """Compute ranking metrics with memory-efficient chunked candidate evaluation.

        all_candidates_cpu: FloatTensor of shape (N, dim) on CPU.
        Candidates are moved to GPU in chunks of self.cand_chunk_size to
        avoid OOM when N is large (e.g. 364K for SNOMED CT).
        """
        query_candidates = self.query_entities[kind]
        query_sentences = Dataset.from_list(query_candidates)["name"]
        # Encode queries on GPU
        query_embeds = model.encode(
            sentences=query_sentences,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )  # (Q, dim) on GPU

        device = query_embeds.device
        Q = query_embeds.size(0)
        N = all_candidates_cpu.size(0)

        if kind in ["nf3", "nf4"]:
            role_sentences = Dataset.from_list(query_candidates)["role"]
            con_sentences = Dataset.from_list(query_candidates)["con"]
            role_embeds = model.tokenizer(role_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
            con_embeds = model.tokenizer(con_sentences, return_tensors="pt", padding=True, truncation=True).to(device)

        answers_id = torch.tensor(self.answer_ids[kind])  # keep on CPU; moved per batch

        all_ranks = []
        iterator = range(0, Q, self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Evaluating {kind}", leave=False)

        for start in iterator:
            end = min(start + self.batch_size, Q)
            answers_id_batch = answers_id[start:end]  # CPU

            if kind in ["nf3", "nf4"] and self.inference_mode == "constructed":
                batch_role_embeds = {k: v[start:end] for k, v in role_embeds.items()}
                batch_con_embeds = {k: v[start:end] for k, v in con_embeds.items()}
                with torch.no_grad():
                    batch_query_embeds = self.ont_model.existence_emb(
                        [batch_role_embeds, batch_con_embeds]
                    ).unsqueeze(1)  # (B, 1, dim)
            else:
                batch_query_embeds = query_embeds[start:end].unsqueeze(1)  # (B, 1, dim)

            # --- Chunked scoring over all N candidates ---------------------------
            # Moving chunks of candidates to GPU avoids allocating (B, N, dim)
            # intermediates that cause OOM for large ontologies.
            chunk_scores_list = []
            for c_start in range(0, N, self.cand_chunk_size):
                c_end = min(c_start + self.cand_chunk_size, N)
                # (1, chunk, dim)
                cand_chunk = all_candidates_cpu[c_start:c_end].unsqueeze(0).to(device)
                if kind == "nf3":
                    scores = self.inference(
                        model, centri_weight,
                        child_embeds=cand_chunk,
                        parent_embeds=batch_query_embeds,
                    )  # (B, chunk)
                else:
                    scores = self.inference(
                        model, centri_weight,
                        child_embeds=batch_query_embeds,
                        parent_embeds=cand_chunk,
                    )  # (B, chunk)
                chunk_scores_list.append((-scores).cpu())
            # ---------------------------------------------------------------------

            predictions = torch.cat(chunk_scores_list, dim=-1)  # (B, N) on CPU
            all_ranks.append(dists_to_ranks(predictions, answers_id_batch))

        ranks = torch.cat(all_ranks, dim=0)
        H1, H10, H100, MRR, MR, median, AUC = compute_metrics(ranks, len(self.all_entities))
        Ranking = RankingResult(H1.item(), H10.item(), H100.item(), ranks.tolist(), AUC)
        return H1, H10, H100, MRR, MR, median, AUC, Ranking

    def __call__(self, model, output_path=None, inference_mode="sentence", epoch=-1, steps=-1, best_centri_weight=None):
        self.inference_mode = inference_mode
        # Encode all candidate concepts on GPU then immediately move to CPU.
        # Chunked scoring in calculate_metrics moves slices back to GPU as needed,
        # keeping peak GPU usage to O(batch × chunk × dim) instead of O(batch × N × dim).
        all_candidates = model.encode(
            sentences=self.all_entities, convert_to_tensor=True, show_progress_bar=False
        ).cpu()  # (N, dim) on CPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if isinstance(best_centri_weight, float):
            if output_path and os.path.exists(os.path.join(output_path, "results.tsv")):
                self.results = pd.read_csv(os.path.join(output_path, "results.tsv"), sep="\t", index_col=0)
            else:
                if output_path:
                    warnings.warn("No previous `results.tsv` detected.")

            all_ranks = []
            for k in ["nf1", "nf2", "nf3", "nf4"]:
                if k not in self.query_entities or len(self.query_entities[k]) == 0:
                    continue
                H1, H10, H100, MRR, MR, median, AUC, ranks = self.calculate_metrics(
                    all_candidates, model, best_centri_weight, kind=k, show_progress=True
                )
                all_ranks.append(ranks)
                best_results = {
                    "axiom_kind": k,
                    "centri_weight": best_centri_weight,
                    "H@1": H1 / len(ranks),
                    "H@10": H10 / len(ranks),
                    "H@100": H100 / len(ranks),
                    "MRR": MRR,
                    "MR": MR,
                    "median": median,
                    "AUC": AUC,
                }
                logger.info(f"Eval results {k}: {best_results}")
                self.results.loc[f"{inference_mode}_{k}"] = best_results
            else:
                all_ranks_combined = combine_rankings(all_ranks, len(self.all_entities))
                best_results = all_ranks_combined.to_dict("combined", best_centri_weight)
                logger.info(f"Combined eval results: {best_results}")
                self.results.loc[f"{inference_mode}_combined"] = best_results
        else:
            self.best_MRR = float("-inf")
            val_kind = "nf1"
            for _k in ["nf1", "nf3", "nf2", "nf4"]:
                if _k in self.query_entities and len(self.query_entities[_k]) > 0:
                    val_kind = _k
                    break
            for centri_weight in range(20):
                centri_weight = centri_weight / 10
                H1, H10, H100, MRR, MR, median, AUC, ranks = self.calculate_metrics(
                    all_candidates, model, centri_weight, kind=val_kind, show_progress=False
                )
                if MRR > self.best_MRR:
                    self.best_MRR = MRR
                    self.best_centri_weight = centri_weight
                    best_results = {
                        "axiom_kind": val_kind,
                        "centri_weight": self.best_centri_weight,
                        "H@1": H1 / len(ranks),
                        "H@10": H10 / len(ranks),
                        "H@100": H100 / len(ranks),
                        "MRR": MRR,
                        "MR": MR,
                        "median": median,
                        "AUC": AUC,
                    }

            idx = f"epoch={epoch}" if epoch != "validation" else epoch
            self.results.loc[idx] = best_results
            logger.info(f"Eval results: {best_results}")

        if output_path:
            os.makedirs(output_path, exist_ok=True)
            self.results.to_csv(os.path.join(output_path, "results.tsv"), sep="\t")

        return best_results
