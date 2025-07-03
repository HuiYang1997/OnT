from __future__ import annotations

from collections import Counter

import logging
import os.path
import warnings

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from .ranking_result import RankingResult, dists_to_ranks, combine_rankings, compute_metrics, compute_rank_roc

logger = logging.getLogger(__name__)


class w2vEvaluator():
    def __init__(self,
        classifier,
        query_entities: dict[str, list[str]],
        answer_ids: dict[str, list[int]],
        all_entities: list[str],
        batch_size: int,
    ):
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
            columns=["axiom_kind", "H@1", "H@10", "H@100", "MRR", "MR", "median", "AUC"]
        )

        # role embedding mode
        self.classifier = classifier


    def calculate_metrics(self, all_candidates, model, kind = 'nf1'):
        query_candidates = self.query_entities[kind]
        query_setences = Dataset.from_list(query_candidates)['name']
        query_embeds = model.encode(sentences=query_setences, batch_size=self.batch_size, convert_to_tensor=True).unsqueeze(1)

        device = query_embeds.device
        
        answers_id = torch.tensor(self.answer_ids[kind]).to(query_embeds.device)
        
        all_ranks = []
        # Process in batches
        for start in tqdm(range(0, query_embeds.size(0), self.batch_size), desc=f"Evaluating {kind}"):
            end = min(start + self.batch_size, query_embeds.size(0))
            answers_id_batch = answers_id[start:end]

            batch_query_embeds = query_embeds[start:end]
            
            # 直接使用PyTorch的广播机制处理张量匹配
            # 1. 将batch_query_embeds扩展到与实体列表相同的大小
            expanded_queries = batch_query_embeds.expand(-1, all_candidates.size(1), -1)
            
            # 2. 将all_candidates扩展到当前批次的大小
            expanded_candidates = all_candidates.expand(batch_query_embeds.size(0), -1, -1)
            
            # 3. 在最后一维进行连接
            concatenated_embeds = torch.cat([expanded_candidates, expanded_queries], dim=2)
            
            # 4. 重新形状为2D张量，分类器要求输入为2D
            batch_size = concatenated_embeds.size(0)
            num_entities = concatenated_embeds.size(1)
            feature_dim = concatenated_embeds.size(2)
            
            # 将[batch_size, num_entities, feature_dim] -> [batch_size*num_entities, feature_dim]
            reshaped_embeds = concatenated_embeds.reshape(-1, feature_dim)
            
            # 转换为numpy并预测
            predictions = self.classifier.predict(reshaped_embeds.cpu().numpy())
            
            # 将预测结果重新形状回[batch_size, num_entities] and convert to tensor
            predictions = torch.tensor(predictions.reshape(batch_size, num_entities)).to(device)
            
            predictions = -1 * predictions
            all_ranks.append(dists_to_ranks(predictions, answers_id_batch))

        # compute H1, H10, H100, MRR, MR, AUC
        # ranks = dists_to_ranks(predictions, answers_id)
        print("computing metrics.......")
        ranks = torch.cat(all_ranks, dim=0)
        H1, H10, H100, MRR, MR, median, AUC = compute_metrics(ranks, len(self.all_entities))

        Ranking = RankingResult(H1.item(), H10.item(), H100.item(), ranks.tolist(), AUC)
        return H1, H10, H100, MRR, MR, median, AUC, Ranking
    


    def __call__(self, model: HierarchyTransformer, output_path: str, task = 'inference'):
        # Call the model and get predictions
        all_candidates = model.encode(sentences=self.all_entities, convert_to_tensor=True).unsqueeze(0)
        
        if task == 'inference':
            nf_list = ['nf1']
        else:
            nf_list = ['nf1', 'nf2', 'nf3', 'nf4']

        all_ranks = []
        for k in nf_list:
            H1, H10, H100, MRR, MR, median, AUC, ranks = self.calculate_metrics(all_candidates, model, kind = k)
            all_ranks.append(ranks)
            # Store results in DataFrame
            best_results = {
                "axiom_kind": k,
                "H@1": H1/len(ranks),
                "H@10": H10/len(ranks),
                "H@100": H100/len(ranks),
                "MRR": MRR,
                "MR": MR,
                "median": median,
                "AUC": AUC,
            }
            logger.info(f"Eval results {k}: {best_results}")
            
            self.results.loc[f"{k}"] = best_results 
        
        # compute all ranks
        all_ranks = combine_rankings(all_ranks, len(self.all_entities))
        combined_results = all_ranks.to_dict("combined", 0)
        logger.info(f"Combined eval results: {combined_results}")
        self.results.loc[f"combined"] = combined_results
        
        self.results.to_csv(os.path.join(output_path, "results.tsv"), sep="\t")
      
        return combined_results
        