"""End-to-end pipeline: OWL → prepare data → train → return model."""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

# Disable noisy progress bars from HuggingFace/safetensors before imports
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from huggingface_hub.utils import disable_progress_bars as _hf_disable
_hf_disable()
import transformers
transformers.utils.logging.disable_progress_bar()
transformers.utils.logging.set_verbosity_error()

import click
from deeponto.utils import set_seed
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from ont.data.load import load_local_dataset
from ont.data.prepare import prepare_ontology_data
from ont.evaluation.evaluator import OnTEvaluator
from ont.hit import HierarchyTransformer
from ont.losses.hit_loss import HierarchyTransformerLoss
from ont.losses.logical_loss import LogicalConstraintLoss
from ont.model import OntologyTransformer
from ont.trainer import HierarchyTransformerTrainer

logger = logging.getLogger(__name__)

# Reduce noise from third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def fit(
    owl_path: str,
    output_dir: str = "./ont_output",
    eval_owl_path: Optional[str] = None,
    test_owl_path: Optional[str] = None,
    balanced: bool = False,
    balanced_negatives: int = 1,
    num_epochs: float = 1,
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
) -> OntologyTransformer:
    """End-to-end: OWL → prepare data → train → return model.

    All axioms from *owl_path* are used for training. 10 % (max 1000) are
    randomly sampled as evaluation data to determine the best lambda.
    No test split is created unless *test_owl_path* is provided.

    If *eval_owl_path* is given, the evaluation data is prepared from that
    ontology instead of sampling from the training ontology.
    If *test_owl_path* is given, test evaluation is performed after training.

    The best lambda (centripetal weight) is determined on the evaluation
    data and saved inside the model so it can be loaded later with
    ``OntologyTransformer.from_pretrained``.

    Args:
        owl_path: Path to OWL/OFN ontology file for training.
        output_dir: Directory for output data and model checkpoints.
        eval_owl_path: Optional separate OWL for evaluation. If None, eval
            data is sampled from the training ontology.
        test_owl_path: Optional separate OWL for testing.
        balanced: Use balanced training with extra C_neg contrastive loss.
        balanced_negatives: Number of negative concept samples for balanced mode.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        learning_rate: Learning rate.
        base_model: Pretrained model name or path.
        role_emd_mode: Role embedding mode.
        role_model_mode: Role model mode.
        existence_loss_kind: Existence loss type.
        clustering_loss_weight: Weight for clustering loss.
        clustering_loss_margin: Margin for clustering loss.
        centripetal_loss_weight: Weight for centripetal loss.
        centripetal_loss_margin: Margin for centripetal loss.
        conj_weight: Weight for conjunction loss.
        exist_weight: Weight for existence loss.
        eval_ratio: Fraction of axioms to sample for evaluation. Default 0.1.
        max_eval: Maximum number of evaluation samples. Default 1000.
        seed: Random seed.

    Returns:
        Trained OntologyTransformer model.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Prepare training data from OWL (all axioms → train)
    # Skip if data directory already fully populated (allows resuming crashed runs).
    data_dir = os.path.join(output_dir, "data")
    _data_ready = all(
        os.path.exists(os.path.join(data_dir, f))
        for f in ("train.jsonl", "concept_names.json", "role_names.json", "val.json")
    )
    if _data_ready:
        logger.info(f"Reusing existing training data from {data_dir}")
    else:
        logger.info(f"Preparing training data from {owl_path} → {data_dir}")
        prepare_ontology_data(owl_path, data_dir, eval_ratio=eval_ratio, max_eval=max_eval)

    # Step 2: Load dataset
    logger.info(f"Loading dataset from {data_dir}")
    dataset = load_local_dataset(data_dir)

    # Step 2b: If external eval ontology provided, prepare it and override val
    if eval_owl_path is not None:
        eval_data_dir = os.path.join(output_dir, "eval_data")
        logger.info(f"Preparing external eval data from {eval_owl_path}")
        prepare_ontology_data(eval_owl_path, eval_data_dir, eval_ratio=1.0, max_eval=max_eval)
        eval_dataset = load_local_dataset(eval_data_dir)
        dataset["val"] = eval_dataset["val"]
        # Use eval ontology concept names for ranking if available
        if len(eval_dataset["concept_names"]) > 0:
            dataset["concept_names"] = eval_dataset["concept_names"]

    # Step 3: Create model (force GPU usage)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading base model: {base_model} on device: {device}")
    hit_model = HierarchyTransformer.from_pretrained(model_name_or_path=base_model, device=device)
    model = OntologyTransformer(hit_model, role_emd_mode=role_emd_mode, role_model_mode=role_model_mode)

    # Step 4: Setup losses
    hit_loss = HierarchyTransformerLoss(
        model=model.hit_model,
        clustering_loss_weight=clustering_loss_weight,
        clustering_loss_margin=clustering_loss_margin,
        centripetal_loss_weight=centripetal_loss_weight,
        centripetal_loss_margin=centripetal_loss_margin,
    )

    all_concept_names = (
        [item["name"] for item in dataset["concept_names"]]
        if len(dataset["concept_names"]) > 0
        else []
    )

    logical_loss = LogicalConstraintLoss(
        model=model,
        hit_loss=hit_loss,
        batch_size=batch_size,
        data_exist=dataset["train_exist"],
        data_conj=dataset["train_conj"],
        conj_weight=conj_weight,
        exist_weight=exist_weight,
        existence_loss_kind=existence_loss_kind,
        balanced=balanced,
        balanced_negatives=balanced_negatives,
        all_concepts=all_concept_names,
    )

    # Step 5: Setup evaluator (may be empty for tiny ontologies)
    has_eval = any(len(dataset["val"]["query_sentences"].get(k, [])) > 0 for k in ["nf1", "nf2", "nf3", "nf4"])
    val_evaluator = None
    if has_eval:
        val_evaluator = OnTEvaluator(
            ont_model=model,
            query_entities=dataset["val"]["query_sentences"],
            answer_ids=dataset["val"]["answer_ids"],
            all_entities=dataset["concept_names"]["name"],
            batch_size=eval_batch_size,
        )

    # Step 6: Training arguments
    # For fractional epochs, eval/save per epoch doesn't make sense.
    _full_epochs = int(num_epochs)
    _is_fractional = num_epochs < 1.0
    experiment_dir = os.path.join(output_dir, "experiment")
    args = SentenceTransformerTrainingArguments(
        output_dir=experiment_dir,
        num_train_epochs=float(num_epochs),
        learning_rate=float(learning_rate),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(eval_batch_size),
        warmup_steps=min(500, max(0, int(0.1 * num_epochs * 342))),  # scale with epochs
        eval_strategy="no" if _is_fractional or not val_evaluator else "epoch",
        save_strategy="no" if _is_fractional else "epoch",
        save_total_limit=2 if not _is_fractional else 0,
        logging_steps=100,
        metric_for_best_model="MRR" if val_evaluator and not _is_fractional else None,
        greater_is_better=True if val_evaluator and not _is_fractional else None,
        load_best_model_at_end=True if val_evaluator and not _is_fractional else False,
        disable_tqdm=False,
        report_to="none",
        use_cpu=False,  # Use GPU if available
        dataloader_pin_memory=False,  # Disable pin_memory to avoid warnings
    )

    # Step 7: Train
    logger.info("Starting training...")
    trainer = HierarchyTransformerTrainer(
        model=model.hit_model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=None,
        loss=logical_loss,
        evaluator=val_evaluator,
    )
    trainer.train()

    # Step 8: Determine best lambda from evaluation data
    best_lambda = 0.0
    if val_evaluator is not None and not val_evaluator.results.empty:
        best_lambda = float(val_evaluator.results.iloc[-1]["centri_weight"])
        logger.info(f"Best lambda (centripetal weight): {best_lambda}")
    else:
        logger.warning("No evaluation data available; using default lambda=0.0")

    model.best_lambda = best_lambda

    val_summary = {"best_lambda": best_lambda}
    with open(os.path.join(experiment_dir, "best_lambda.json"), "w") as f:
        json.dump(val_summary, f, indent=2)

    # Step 8b: Optional test evaluation on external test ontology
    if test_owl_path is not None:
        test_data_dir = os.path.join(output_dir, "test_data")
        logger.info(f"Preparing test data from {test_owl_path}")
        prepare_ontology_data(test_owl_path, test_data_dir, eval_ratio=1.0, max_eval=100000)
        test_dataset = load_local_dataset(test_data_dir)
        test_concept_names = test_dataset["concept_names"]["name"] if len(test_dataset["concept_names"]) > 0 else dataset["concept_names"]["name"]
        has_test = any(len(test_dataset["val"]["query_sentences"].get(k, [])) > 0 for k in ["nf1", "nf2", "nf3", "nf4"])
        if has_test:
            test_evaluator = OnTEvaluator(
                ont_model=model,
                query_entities=test_dataset["val"]["query_sentences"],
                answer_ids=test_dataset["val"]["answer_ids"],
                all_entities=test_concept_names,
                batch_size=eval_batch_size,
            )
            test_result = test_evaluator(
                model=model.hit_model,
                output_path=os.path.join(experiment_dir, "test_eval"),
                best_centri_weight=best_lambda,
                inference_mode="sentence",
            )
            logger.info(f"Test results: {test_result}")

    # Step 9: Save final model (with best_lambda baked in)
    final_dir = os.path.join(output_dir, "final")
    model.save(final_dir)
    logger.info(f"Model saved to {final_dir}")

    return model


# ------------------------------------------------------------------ #
#  CLI entry point
# ------------------------------------------------------------------ #

@click.command()
@click.option("--owl", required=True, type=click.Path(exists=True), help="Path to OWL/OFN ontology file.")
@click.option("--output", default="./ont_output", help="Output directory.")
@click.option("--eval-owl", default=None, type=click.Path(exists=True), help="Optional OWL for evaluation.")
@click.option("--test-owl", default=None, type=click.Path(exists=True), help="Optional OWL for testing.")
@click.option("--epochs", default=1, type=int, help="Number of training epochs.")
@click.option("--batch-size", default=64, type=int, help="Training batch size (sentences per step).")
@click.option("--eval-batch-size", default=32, type=int, help="Evaluation batch size (queries scored per step).")
@click.option("--lr", default=1e-5, type=float, help="Learning rate.")
@click.option("--base-model", default="sentence-transformers/all-MiniLM-L12-v2", help="Pretrained model.")
@click.option("--balanced/--no-balanced", default=False, help="Use balanced training mode.")
@click.option("--balanced-negatives", default=1, type=int, help="Number of balanced negatives.")
@click.option("--seed", default=8888, type=int, help="Random seed.")
def cli(owl, output, eval_owl, test_owl, epochs, batch_size, eval_batch_size, lr, base_model, balanced, balanced_negatives, seed):
    """Train an OntologyTransformer from an OWL ontology file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    fit(
        owl_path=owl,
        output_dir=output,
        eval_owl_path=eval_owl,
        test_owl_path=test_owl,
        num_epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=lr,
        base_model=base_model,
        balanced=balanced,
        balanced_negatives=balanced_negatives,
        seed=seed,
    )


if __name__ == "__main__":
    cli()
