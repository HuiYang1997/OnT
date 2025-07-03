# Copyright 2023 Yuan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This training script is hierarchy re-training of HiT models."""
from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime

import click
from deeponto.utils import create_path, load_file, set_seed
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from yacs.config import CfgNode

from src.hierarchy_transformers.datasets import load_hf_dataset, load_local_dataset
from src.hierarchy_transformers.evaluation import HierarchyTransformerEvaluator
from src.hierarchy_transformers.evaluation import OnTEvaluator
from src.hierarchy_transformers.losses import HierarchyTransformerLoss, LogicalConstraintLoss
from src.hierarchy_transformers.models.hierarchy_transformer import OntologyTransformer, HierarchyTransformer, HierarchyTransformerTrainer

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
def main(config_file: str):
    # 0. set seed, load config, and format output dir
    set_seed(8888)
    config = CfgNode(load_file(config_file))
    model_path_suffix = config.model_path.split(os.path.sep)[-1]
    output_dir = f"experiments/OnTr-{model_path_suffix}-{config.dataset_name}"
    create_path(output_dir)
    try:
        shutil.copy2(config_file, os.path.join(output_dir, "config.yaml"))
    except Exception:
        pass

    # 1. Load dataset and pre-trained model
    # NOTE: according to docs, it is very important to have column names ["child", "parent", "negative"] *in order* to match ["anchor", "positive", "negative"]
    our_dataset = load_local_dataset(config.dataset_path)
    base_model = HierarchyTransformer.from_pretrained(model_name_or_path=config.model_path)
    model = OntologyTransformer(base_model, role_emd_mode=config.role_emd_mode, role_model_mode=config.role_model_mode)

    # 2. set up the loss function
    hit_loss = HierarchyTransformerLoss(
        model=model.hit_model,
        clustering_loss_weight=config.hit_loss.clustering_loss_weight,
        clustering_loss_margin=config.hit_loss.clustering_loss_margin,
        centripetal_loss_weight=config.hit_loss.centripetal_loss_weight,
        centripetal_loss_margin=config.hit_loss.centripetal_loss_margin,
    )
    logger.info(f"OnT loss config: {hit_loss.get_config_dict()}")

    # Create the logical constraint loss
    logical_loss = LogicalConstraintLoss(
        model=model,
        hit_loss=hit_loss,
        batch_size=config.train_batch_size,
        data_exist=our_dataset["train_exist"],
        data_conj=our_dataset["train_conj"],
        conj_weight=config.logical_loss.conj_weight,
        exist_weight=config.logical_loss.exist_weight,
        existence_loss_kind=config.existence_loss_kind,
        # role_inverse=our_dataset["role_inverse"],
    )
    logger.info(f"Logical constraint loss config: {logical_loss.get_config_dict()}")

    # 3. Define a validation evaluator for use during training.
    val_evaluator = OnTEvaluator(
        ont_model=model,
        query_entities = our_dataset["val"]['query_sentences'],
        answer_ids = our_dataset["val"]['answer_ids'],
        all_entities = our_dataset['concept_names']['name'],
        batch_size=config.eval_batch_size,
    )

    # 4. Define the training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        # max_steps = 1,
        num_train_epochs=int(config.num_train_epochs),
        learning_rate=float(config.learning_rate),
        per_device_train_batch_size=int(config.train_batch_size),
        per_device_eval_batch_size=int(config.eval_batch_size),
        warmup_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=100,
        metric_for_best_model="MRR",  # to override loss for model selection
        greater_is_better=True,  # due to MRR score
        load_best_model_at_end=True,
    )

    # 5. Create the trainer & start training with multiple datasets and losses
    trainer = HierarchyTransformerTrainer(
        model=model.hit_model,
        args=args,
        train_dataset=our_dataset["train"],
        eval_dataset=None,
        loss=logical_loss,  # Use HiT loss and logical constraint loss
        evaluator=val_evaluator,
    )
    trainer.train()

    # 6. Evaluate the model performance on the test dataset
    val_results = val_evaluator.results
    best_val_centri_weight = float(val_results.iloc[-1]["centri_weight"])
    print(best_val_centri_weight)
    # best_val_centri_weight = 0.0
    test_evaluator = OnTEvaluator(
        ont_model=model,
        query_entities = our_dataset["test"]['query_sentences'],
        answer_ids = our_dataset["test"]['answer_ids'],
        all_entities = our_dataset['concept_names']['name'],
        batch_size=config.eval_batch_size,
    )
    for inference_mode in ["sentence"]:
        result = test_evaluator(
            model=model.hit_model,
            output_path=os.path.join(output_dir, "eval"),
            best_centri_weight=best_val_centri_weight,
            inference_mode=inference_mode,
        )
        log_results(result, config, inference_mode)

    # 7. Save the trained & evaluated model locally
    final_output_dir = f"{output_dir}/final"
    model.save(final_output_dir)


def log_results(result, config, inference_mode):
    """save results and hyperparameters to a log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file =  "training_log.txt"
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Inference Mode: {inference_mode}\n\n")
        
        # write hyperparameters
        f.write("Hyperparameters:\n")
        f.write(f"Model: {config.model_path}\n")
        f.write(f"Dataset: {config.dataset_name}\n")
        f.write(f"Epochs: {config.num_train_epochs}\n")
        f.write(f"Learning Rate: {config.learning_rate}\n")
        f.write(f"Role Embedding Mode: {config.role_emd_mode}\n")
        f.write(f"Role Model Mode: {config.role_model_mode}\n")
        f.write(f"Existence Loss Kind: {config.existence_loss_kind}\n")
        
        # write results
        f.write("\nResults:\n")
        for metric, value in result.items():
            f.write(f"{metric}: {value}\n")
        f.write("\n")


if __name__ == "__main__":
    main()
