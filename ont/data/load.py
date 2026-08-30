from __future__ import annotations

import json
import logging
import os

from datasets import Dataset

logger = logging.getLogger(__name__)


def load_local_dataset(path: str) -> dict:
    """Load a local OnT dataset from a directory.

    Expected files:
        - train.jsonl: training triplets (child, parent, negative)
        - train_conj.jsonl: conjunction pairs (Concept, con1, con2)
        - train_exist.jsonl: existential pairs (Concept, role, con)
        - val.json: validation data
        - test.json: test data
        - concept_names.json: id → concept name mapping
        - role_names.json: id → role name mapping
        - role_inverse.json: role → inverse role mapping

    Returns:
        Dictionary with keys: train, train_conj, train_exist, val, test, concept_names, role_inverse
    """
    datafiles = dict()

    _empty_eval = {"query_sentences": {"nf1": [], "nf2": [], "nf3": [], "nf4": []},
                    "answer_ids": {"nf1": [], "nf2": [], "nf3": [], "nf4": []}}

    for split in ["train", "train_conj", "train_exist", "val", "test", "concept_names", "role_inverse"]:
        if split in ["train", "train_conj", "train_exist"]:
            split_path = os.path.join(path, f"{split}.jsonl")
            if not os.path.exists(split_path):
                logger.warning(f"File not found: {split_path}, using empty dataset for {split}")
                datafiles[split] = Dataset.from_list([])
                continue
            split_list = []
            with open(split_path, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    current_dict = json.loads(line)
                    if split == "train":
                        child = current_dict["child"]
                        parent = current_dict["parent"]
                        neg_list = current_dict["negative"]
                        split_list += [{"child": child, "parent": parent, "negative": neg} for neg in neg_list[:1]]
                    else:
                        split_list.append(current_dict)
            datafiles[split] = Dataset.from_list(split_list)
        else:
            split_path = os.path.join(path, f"{split}.json")
            if not os.path.exists(split_path):
                if split == "role_inverse":
                    datafiles[split] = Dataset.from_list([])
                elif split == "concept_names":
                    datafiles[split] = Dataset.from_list([])
                elif split in ("val", "test"):
                    datafiles[split] = _empty_eval.copy()
                else:
                    datafiles[split] = {}
                if split != "test":
                    logger.warning(f"File not found: {split_path}, using empty data for {split}")
                continue
            with open(split_path, "r") as f:
                dataset = json.load(f)
            if split == "concept_names":
                num_concept = len(dataset)
                data_list = [{"name": dataset[str(i)]} for i in range(num_concept)]
                datafiles[split] = Dataset.from_list(data_list)
            elif split == "role_inverse":
                data_list = [{"role": k, "inverse": dataset[k]} for k in dataset.keys()]
                datafiles[split] = Dataset.from_list(data_list)
            else:
                datafiles[split] = dataset

    return datafiles
