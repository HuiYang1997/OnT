from __future__ import annotations

from typing import Any

import torch
from sentence_transformers.trainer import SentenceTransformerTrainer

from ont.hit import HierarchyTransformer


class HierarchyTransformerTrainer(SentenceTransformerTrainer):
    r"""Extension of SentenceTransformerTrainer to monitor and log batch losses of HierarchyTransformer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-set widget to prevent "Computing widget examples" progress bar
        if hasattr(self.model, "model_card_data"):
            self.model.model_card_data.widget = [{"text": "example"}]
        # Remove model card callback
        self.callback_handler.callbacks = [
            cb for cb in self.callback_handler.callbacks
            if type(cb).__name__ != "SentenceTransformerModelCardCallback"
        ]

    def compute_loss(
        self,
        model: HierarchyTransformer,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        loss_dict = super().compute_loss(
            model=model, inputs=inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )
        outputs = None
        if return_outputs:
            loss_dict, outputs = loss_dict

        # loss_dict may be a plain tensor (from base trainer) or a dict (from our losses)
        if isinstance(loss_dict, dict):
            if "conj_loss" in loss_dict:
                self.log(
                    {
                        "conj_loss": round(loss_dict["conj_loss"].item(), 4),
                        "exist_loss": round(loss_dict["exist_loss"].item(), 4),
                        "cluster_loss": round(loss_dict["cluster_loss"].item(), 4),
                        "centri_loss": round(loss_dict["centri_loss"].item(), 4),
                        "combined_loss": round(loss_dict["loss"].item(), 4),
                    }
                )
            else:
                self.log(
                    {
                        "cluster_loss": round(loss_dict["cluster_loss"].item(), 4),
                        "centri_loss": round(loss_dict["centri_loss"].item(), 4),
                        "combined_loss": round(loss_dict["loss"].item(), 4),
                    }
                )
            loss = loss_dict["loss"]
        else:
            loss = loss_dict

        return (loss, outputs) if return_outputs else loss
