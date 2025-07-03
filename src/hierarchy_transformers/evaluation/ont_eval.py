from __future__ import annotations

from collections import Counter

import logging
import os.path
import warnings

import numpy as np
import pandas as pd
import torch
from sentence_transformers.evaluation import SentenceEvaluator

from datasets import Dataset
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from hierarchy_transformers.evaluation.metrics import evaluate_by_threshold, grid_search
from .ranking_result import RankingResult, dists_to_ranks, combine_rankings, compute_metrics, compute_rank_roc

logger = logging.getLogger(__name__)


class OnTEvaluator(SentenceEvaluator):
    def __init__(self,
        ont_model,
        query_entities: dict[str, list[str]],
        answer_ids: dict[str, list[int]],
        all_entities: list[str],
        batch_size: int,
    ):
        super().__init__()
        # set primary metric for model selection
        self.primary_metric = "H@1"
        # input evaluation examples
        self.query_entities = query_entities
        self.answer_ids = answer_ids
        self.all_entities = all_entities
        # eval batch size
        self.batch_size = batch_size
        # result file
        self.results = pd.DataFrame(
            columns=["axiom_kind", "centri_weight", "H@1", "H@10", "H@100", "MRR", "MR", "median", "AUC"]
        )

        # role embedding mode
        self.ont_model = ont_model
        self.inference_mode = 'sentence'
    
    def inference(
        self,
        model: HierarchyTransformer,
        centri_weight: float,
        child_embeds: torch.Tensor | None = None,
        parent_embeds: torch.Tensor | None = None,
    ):
        """The default probing method of the HiT model. It output scores that indicate hierarchical relationships between entities.

        Optional `child_embeds` and `parent_embeds` are used to save time from repetitive encoding.
        """
        dists = model.manifold.dist(child_embeds, parent_embeds)
        child_norms = model.manifold.dist0(child_embeds)
        parent_norms = model.manifold.dist0(parent_embeds)
        return -(dists + centri_weight * (parent_norms - child_norms)) 
    


    def calculate_metrics(self, all_candidates, model, centri_weight, kind = 'nf1'):
        query_candidates = self.query_entities[kind]
        query_setences = Dataset.from_list(query_candidates)['name']
        query_embeds = model.encode(sentences=query_setences, batch_size=self.batch_size, convert_to_tensor=True).unsqueeze(1)

        device = query_embeds.device
        if kind == 'nf2':
            con1_setences = Dataset.from_list(query_candidates)['con1']
            con2_setences = Dataset.from_list(query_candidates)['con2']
            con1_embeds=model.tokenizer(con1_setences, return_tensors="pt", padding=True, truncation=True).to(device)
            con2_embeds=model.tokenizer(con2_setences, return_tensors="pt", padding=True, truncation=True).to(device)  
        elif kind == 'nf3' or kind == 'nf4':
            role_sentences = Dataset.from_list(query_candidates)['role']
            con_sentences = Dataset.from_list(query_candidates)['con']
            role_embeds=model.tokenizer(role_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
            con_embeds=model.tokenizer(con_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
        
        answers_id = torch.tensor(self.answer_ids[kind]).to(query_embeds.device)
        
        # Initialize a list to store predictions
        # all_predictions = []
        all_ranks = []
        # Process in batches
        for start in tqdm(range(0, query_embeds.size(0), self.batch_size), desc=f"Evaluating {kind}"):
            end = min(start + self.batch_size, query_embeds.size(0))
            answers_id_batch = answers_id[start:end]

            if kind in ['nf3', 'nf4'] and self.inference_mode == 'constructed':
                batch_role_embeds = {k: v[start:end] for k, v in role_embeds.items()}
                batch_con_embeds = {k: v[start:end] for k, v in con_embeds.items()}
                with torch.no_grad():
                    batch_query_embeds = self.ont_model.existence_emb([batch_role_embeds, batch_con_embeds]).unsqueeze(1)
            else:
                batch_query_embeds = query_embeds[start:end]

            if kind == 'nf3':
                predictions = self.inference(model, centri_weight, child_embeds=all_candidates, parent_embeds=batch_query_embeds)
            else:
                predictions = self.inference(model, centri_weight, child_embeds=batch_query_embeds, parent_embeds=all_candidates)
            
            predictions = -1 * predictions
            all_ranks.append(dists_to_ranks(predictions, answers_id_batch))
            # all_predictions.append(predictions)


        # # Concatenate all predictions
        # predictions = torch.cat(all_predictions, dim=0)
        # print("all_predictions[-1].shape, predictions.shape", all_predictions[-1].shape, predictions.shape)
        # predictions = -1 * predictions

        # if kind == 'nf1':
        #     # replace 0 with inf for filter c <= c
        #     # when computing the metrics we ignore the negative sybmols, therefore multiply the predictions by -1
        #     predictions = -1 * torch.where(predictions == 0, torch.tensor(float('-inf')), predictions)
        # else:
        #     predictions = -1 * predictions

        # compute H1, H10, H100, MRR, MR, AUC
        # ranks = dists_to_ranks(predictions, answers_id)
        print("computing metrics.......")
        ranks = torch.cat(all_ranks, dim=0)
        H1, H10, H100, MRR, MR, median, AUC = compute_metrics(ranks, len(self.all_entities))

        Ranking = RankingResult(H1.item(), H10.item(), H100.item(), ranks.tolist(), AUC)
        return H1, H10, H100, MRR, MR, median, AUC, Ranking
    


    def __call__(self, model: HierarchyTransformer, output_path: str | None = None, inference_mode: str = 'sentence', epoch: int = -1, steps: int = -1, best_centri_weight: float | None = None):
        # Call the model and get predictions
        self.inference_mode = inference_mode
        all_candidates = model.encode(sentences=self.all_entities, convert_to_tensor=True).unsqueeze(0)
        if  isinstance(best_centri_weight, float):
            # log the results
            if os.path.exists(os.path.join(output_path, "results.tsv")):
                self.results = pd.read_csv(os.path.join(output_path, "results.tsv"), sep="\t", index_col=0)
            else:
                warnings.warn("No previous `results.tsv` detected.")
            
            all_ranks = []
            for k in ['nf1', 'nf2', 'nf3', 'nf4']:
                if k not in self.query_entities:
                    break
                H1, H10, H100, MRR, MR, median, AUC, ranks = self.calculate_metrics(all_candidates, model, best_centri_weight, kind = k)
                all_ranks.append(ranks)
                # Store results in DataFrame
                best_results = {
                    "axiom_kind": k,
                    "centri_weight": best_centri_weight,
                    "H@1": H1/len(ranks),
                    "H@10": H10/len(ranks),
                    "H@100": H100/len(ranks),
                    "MRR": MRR,
                    "MR": MR,
                    "median": median,
                    "AUC": AUC,
                }
                logger.info(f"Eval results {k}: {best_results}")
                
                self.results.loc[f"{inference_mode}_{k}"] = best_results
            else:
                # compute all ranks
                all_ranks = combine_rankings(all_ranks, len(self.all_entities))
                best_results = all_ranks.to_dict("combined", best_centri_weight)
                logger.info(f"Combined eval results: {best_results}")
                self.results.loc[f"{inference_mode}_combined"] = best_results
        else:
            self.best_MRR = float('-inf')  # Initialize best Mean Rank to infinity for minimization
            for centri_weight in range(20):
                centri_weight = centri_weight / 10
                H1, H10, H100, MRR, MR, median, AUC, ranks = self.calculate_metrics(all_candidates, model, centri_weight, kind = 'nf1') 
                if MRR > self.best_MRR:
                    self.best_MRR = MRR
                    self.best_centri_weight = centri_weight
                    best_results = {
                        "axiom_kind": 'nf1',
                        "centri_weight": self.best_centri_weight,
                        "H@1": H1/len(ranks),
                        "H@10": H10/len(ranks),
                        "H@100": H100/len(ranks),
                        "MRR": MRR,
                        "MR": MR,
                        "median": median,
                        "AUC": AUC,
                    }
            
            idx = f"epoch={epoch}" if epoch != "validation" else epoch
            self.results.loc[idx] = best_results
            logger.info(f"Eval results: {best_results}")
      
        self.results.to_csv(os.path.join(output_path, "results.tsv"), sep="\t")
      
        return best_results
        