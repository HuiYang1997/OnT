from __future__ import annotations

import math

import torch
from tqdm import tqdm


def f1_score(predictions: torch.Tensor, labels: torch.Tensor, truth_label: int = 1):
    """Compute F1 score from predictions and labels."""
    tp = torch.sum((labels == truth_label) & (predictions == truth_label))
    fp = torch.sum((labels != truth_label) & (predictions == truth_label))
    fn = torch.sum((labels == truth_label) & (predictions != truth_label))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision.item(), "recall": recall.item(), "f1": f1.item()}


def accuracy(predictions: torch.Tensor, labels: torch.Tensor):
    """Compute accuracy."""
    acc = torch.sum(labels == predictions) / len(labels)
    return {"accuracy": acc.item()}


def accuracy_on_negatives(predictions: torch.Tensor, labels: torch.Tensor, truth_label: int = 1):
    """Compute accuracy only on negative samples."""
    neg_acc = torch.sum((labels == predictions) & (labels != truth_label)) / torch.sum(labels != truth_label)
    return {"accuracy_on_negatives": neg_acc.item()}


def evaluate_by_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    truth_label: int = 1,
    smaller_scores_better: bool = False,
):
    """Compute evaluation metrics based on threshold."""
    if smaller_scores_better:
        predictions = scores <= threshold
    else:
        predictions = scores > threshold
    results = {
        "threshold": threshold,
        **f1_score(predictions=predictions, labels=labels, truth_label=truth_label),
        **accuracy(predictions=predictions, labels=labels),
        **accuracy_on_negatives(predictions=predictions, labels=labels, truth_label=truth_label),
    }
    return results


def grid_search(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold_granularity: int = 100,
    truth_label: int = 1,
    smaller_scores_better: bool = False,
    primary_metric: str = "f1",
    best_primary_metric_value: float = -math.inf,
    preformatted_best_results: dict = {},
):
    """Grid search the best scoring threshold."""
    best_results = None
    start = int(scores.min() * threshold_granularity)
    end = int(scores.max() * threshold_granularity)

    for threshold in tqdm(range(start, end), desc="Thresholding"):
        threshold = threshold / threshold_granularity
        results = evaluate_by_threshold(
            scores=scores,
            labels=labels,
            threshold=threshold,
            truth_label=truth_label,
            smaller_scores_better=smaller_scores_better,
        )
        if results[primary_metric] >= best_primary_metric_value:
            best_results = preformatted_best_results
            best_results.update(results)
            best_primary_metric_value = results[primary_metric]

    return best_results
