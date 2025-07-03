from dataclasses import dataclass
import numpy as np

from typing import List

import torch

from collections import Counter


def compute_metrics(ranks, num_entities):
    ranks = ranks.cpu().numpy()


    H1 = (ranks == 1).sum() 
    H10 = (ranks <= 10).sum()
    H100 = (ranks <= 100).sum() 
    print(f"H1: {H1}, H10: {H10}, H100: {H100}")
    MRR = np.mean([1 / r for r in ranks])
    MR = np.mean(ranks)
    print(f"MRR: {MRR}, MR: {MR}")
    median = np.median(ranks)
    rank_dict = Counter(ranks)
    AUC = compute_rank_roc(rank_dict, num_entities)
    return H1, H10, H100, MRR, MR, median, AUC

def compute_rank_roc(ranks_dict, num_classes):
    sorted_ranks = sorted(list(ranks_dict.keys()))
    tprs = [0]
    fprs = [0]
    tpr = 0
    num_triples = sum(ranks_dict.values())
    num_negatives = (num_classes - 1) * num_triples
    for x in sorted_ranks:
        tpr += ranks_dict[x]
        tprs.append(tpr / num_triples)
        fp = sum([(x - 1) * v if k <= x else x * v for k, v in ranks_dict.items()])
        fprs.append(fp / num_negatives)

    tprs.append(1)
    fprs.append(1)
    auc = np.trapz(tprs, fprs)
    return auc
    
def dists_to_ranks(dists, targets):
    index = torch.argsort(dists, dim=1).argsort(dim=1) + 1
    return torch.take_along_dim(index, targets.reshape(-1, 1), dim=1).flatten()

def combine_rankings(rankings, num_classes):
    combined_ranking = RankingResult(0, 0, 0, [], 0)
    for ranking in rankings:
        combined_ranking = combined_ranking.combine(ranking)
    ranks_dict = Counter(combined_ranking.ranks)
    auc = compute_rank_roc(ranks_dict, num_classes)
    combined_ranking.auc = auc
    return combined_ranking

@dataclass
class RankingResult:
    top1: int
    top10: int
    top100: int
    ranks: List[int]
    auc: float

    def combine(self, other):
        return RankingResult(self.top1 + other.top1,
                             self.top10 + other.top10,
                             self.top100 + other.top100,
                             self.ranks + other.ranks,
                             0)  # The AUC needs to be recalculated after combining results

    def __str__(self):
        return f'top1: {self.top1 / len(self):.2f}, top10: {self.top10 / len(self):.2f}, ' \
               f'top100: {self.top100 / len(self):.2f}, mean: {round(np.mean(self.ranks))}, median: {round(np.median(self.ranks))}, ' \
               f'mrr: {np.mean([1 / r for r in self.ranks]):.2f}, auc: {self.auc:.2f}'

    def __len__(self):
        return len(self.ranks)

    def to_csv(self):
        return f'{self.top1 / len(self):.2f},{self.top10 / len(self):.2f},' \
               f'{self.top100 / len(self):.2f},{round(np.median(self.ranks))},{np.mean([1 / r for r in self.ranks]):.2f},' \
               f'{round(np.mean(self.ranks))},{self.auc:.2f}'
    
    def to_dict(self, kind, centri_weight):
        return {
                    "axiom_kind": kind,
                    "centri_weight": centri_weight,
                    "H@1": self.top1 / len(self),
                    "H@10": self.top10 / len(self),
                    "H@100": self.top100 / len(self),
                    "MRR": np.mean([1 / r for r in self.ranks]),
                    "MR": np.mean(self.ranks),
                    "median": np.median(self.ranks),
                    "AUC": self.auc,
                }
