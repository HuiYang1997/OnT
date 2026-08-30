"""Prepare OnT training data from an OWL/OFN ontology file.

Flow: OWL → deeponto load → normalize → verbalize → generate OnT training files.
Only produces OnT data (train.jsonl, train_exist.jsonl, train_conj.jsonl, val.json, test.json, etc.)
"""
from __future__ import annotations

import json
import logging
import os
import re
from random import sample
from typing import Dict, List, Optional

import numpy as np
from yacs.config import CfgNode

logger = logging.getLogger(__name__)

# Fix random seed for reproducibility
np.random.seed(42)
import random
random.seed(42)


def _ensure_jvm(memory: str = "8g"):
    """Start JVM for deeponto/jpype before importing deeponto.onto (which prompts interactively)."""
    import jpype
    if not jpype.isJVMStarted():
        from deeponto import init_jvm
        init_jvm(memory)



def camel_case_to_spaced(phrase: str) -> str:
    """Convert camelCase or IRI fragment to spaced phrase."""
    phrase = phrase.split("#")[-1]
    segments = re.findall(r"[A-Z]+|[^A-Z]+", phrase)
    second_phrase = ""
    numbers_romain = {"II", "III", "IV", "V", "VI", "VII", "VIII", "IX"}
    for segment in segments:
        if segment[-1].isupper():
            if len(segment) == 1:
                segment = segment.lower()
            elif len(segments) > 1 and segment not in numbers_romain:
                segment = segment[:-1] + " " + segment[-1].lower()
        else:
            segment = segment + " "
        second_phrase += segment
    return second_phrase.strip()


class ELNormalizedData:
    """Create EL normalized dataset for OnT embeddings.

    Only produces OnT training data (no prediction/ or normalization/ subfolders).
    """

    def __init__(self):
        self.concept_names: Dict[str, tuple] = {}  # name → (ind, iri)
        self.role_names: Dict[str, tuple] = {}      # name → (ind, iri)
        # Only original axioms (no decomposition)
        self.nf1_org: List[list] = []  # A ⊑ B
        self.nf2_org: List[list] = []  # A ⊓ B ⊑ C
        self.nf3_org: List[list] = []  # A ⊑ ∃r.B
        self.nf4_org: List[list] = []  # ∃r.A ⊑ B
        self.role_inclusion: List[list] = []
        self.output_dir = None

    def initial_atomic_names(self, ont, vocab) -> None:
        """Initialize concept and role names from ontology vocabulary."""
        i = 0
        for iri in ont.owl_classes:
            if vocab[iri] in self.concept_names:
                logger.debug(f"{iri} repeat to {self.concept_names[vocab[iri]]}, name: {vocab[iri]}")
            else:
                self.concept_names[vocab[iri]] = (i, iri)
                i += 1
        j = 0
        for iri in ont.owl_object_properties:
            if vocab[iri] in self.role_names:
                logger.debug(f"{iri} repeat to {self.role_names[vocab[iri]]}, name: {vocab[iri]}")
            else:
                self.role_names[vocab[iri]] = (j, iri)
                j += 1

    def update_from_subsumptions(self, child, parent) -> None:
        """Process subsumption axioms - keep original axioms only, no decomposition."""
        if parent["type"] == "AND":
            # Parent is conjunction - process each conjunct separately
            for C_i in parent["classes"]:
                self.update_from_subsumptions(child, C_i)
            return
        elif parent["type"] == "EX.":
            # A ⊑ ∃r.B - nf3
            if child["type"] != "IRI":
                return  # Skip if child is complex
            C_id = self.concept_names[child["verbal"]][0]
            role_name = parent["property"]["verbal"]
            role_id = self.role_names[role_name][0]
            if parent["class"]["type"] != "IRI":
                return  # Skip if filler is complex
            D_id = self.concept_names[parent["class"]["verbal"]][0]
            self.nf3_org.append([C_id, role_id, D_id])
        else:
            # Parent is atomic – skip unsupported types (NOT, OR, DATA, …)
            if parent["type"] != "IRI":
                return
            parent_id = self.concept_names[parent["verbal"]][0]
            if child["type"] == "IRI":
                # A ⊑ B - nf1
                child_id = self.concept_names[child["verbal"]][0]
                self.nf1_org.append([child_id, parent_id])
            elif child["type"] == "AND":
                # A ⊓ B ⊑ C - nf2 (only if both are atomic)
                if len(child["classes"]) == 2:
                    c1, c2 = child["classes"]
                    if c1["type"] == "IRI" and c2["type"] == "IRI":
                        c1_id = self.concept_names[c1["verbal"]][0]
                        c2_id = self.concept_names[c2["verbal"]][0]
                        self.nf2_org.append([c1_id, c2_id, parent_id])
            elif child["type"] == "EX.":
                # ∃r.A ⊑ B - nf4 (only if filler is atomic)
                if child["class"]["type"] == "IRI":
                    filler_id = self.concept_names[child["class"]["verbal"]][0]
                    role_name = child["property"]["verbal"]
                    role_id = self.role_names[role_name][0]
                    self.nf4_org.append([role_id, filler_id, parent_id])

    def update_from_equivalent(self, child, parent) -> None:
        """Process equivalence axioms as bidirectional subsumptions."""
        self.update_from_subsumptions(child, parent)
        self.update_from_subsumptions(parent, child)

    def load(self, ontology_path: str, jvm_memory: str = "8g"):
        """Load ontology and initialize vocabulary."""
        _ensure_jvm(jvm_memory)
        from deeponto.onto import Ontology, OntologyVerbaliser

        ont = Ontology(ontology_path)
        verbalizer = OntologyVerbaliser(ont, add_quantifier_word=True)

        if not verbalizer.vocab:
            logger.warning("Verbalizer vocabulary is empty. Using camelCase vocabulary from IRIs.")
            for iri in ont.owl_classes:
                verbalizer.update_entity_name(iri, camel_case_to_spaced(iri))
            for iri in ont.owl_object_properties:
                verbalizer.update_entity_name(iri, camel_case_to_spaced(iri))

        self.initial_atomic_names(ont, verbalizer.vocab)
        return ont, verbalizer

    def create_dataset(self, ont, verbalizer):
        """Process ontology axioms directly without normalization."""
        # Use getImportsClosure() which returns the ontology itself plus all
        # directly/indirectly imported ontologies — each visited exactly once.
        axioms = []
        for ont_in_closure in ont.owl_onto.getImportsClosure():
            axioms.extend(list(ont_in_closure.getAxioms()))

        for axiom in axioms:
            axiom_type = verbalizer.onto.get_axiom_type(axiom)
            if axiom_type == "SubClassOf":
                try:
                    verb_result = verbalizer.verbalise_class_subsumption_axiom(axiom)
                except Exception:
                    logger.debug(f"Failed to verbalise axiom: {axiom}")
                    continue
                self.update_from_subsumptions(*verb_result)
            elif axiom_type == "EquivalentClasses":
                try:
                    verb_result = verbalizer.verbalise_class_equivalence_axiom(axiom)
                except Exception:
                    logger.debug(f"Failed to verbalise axiom: {axiom}")
                    continue
                self.update_from_equivalent(*verb_result)

    def get_role_inclusion(self, ont, verbalizer):
        """Extract role inclusion axioms."""
        for ont_axiom in ont.get_subsumption_axioms("ObjectProperties"):
            try:
                sub_verbal, sup_verbal = verbalizer.verbalise_object_property_subsumption_axiom(ont_axiom)
            except Exception:
                continue
            sub_id = self.role_names[sub_verbal["verbal"]][0]
            sup_id = self.role_names[sup_verbal["verbal"]][0]
            self.role_inclusion.append([sub_id, sup_id])

    def save_ont_data(self, eval_ratio: float = 0.1, max_eval: int = 1000):
        """Save OnT training data. All axioms go to train. A random subset is
        sampled as evaluation data (10% by default, capped at *max_eval*).
        No test split is produced.
        """
        out_dir = self.output_dir

        concept_names = {str(v[0]): k for k, v in self.concept_names.items()}
        role_names = {str(v[0]): k for k, v in self.role_names.items()}
        concept_list = list(concept_names.values())

        logger.info(f"nf1_org: {len(self.nf1_org)}, nf2_org: {len(self.nf2_org)}, "
                     f"nf3_org: {len(self.nf3_org)}, nf4_org: {len(self.nf4_org)}")

        # Only use original axioms (no decomposition)
        all_nfs = {}
        for i, nf_org in enumerate([self.nf1_org, self.nf2_org, self.nf3_org, self.nf4_org], 1):
            cols = 2 if i <= 2 else 3
            arr_org = np.array(nf_org) if nf_org else np.array([]).reshape(0, cols)
            np.random.shuffle(arr_org)
            all_nfs[f"nf{i}"] = arr_org

        # ---- Convert all axioms to text-based train data ----
        all_train_data = []
        conj_data = []
        exist_data = []
        # Also collect per-nf index lists so we can sample eval proportionally
        nf_indices: Dict[str, List[int]] = {f"nf{i}": [] for i in range(1, 5)}

        for nf_kind in ["nf1", "nf2", "nf3", "nf4"]:
            data = all_nfs[nf_kind]
            if data.size == 0:
                continue
            for item in data:
                idx = len(all_train_data)
                nf_indices[nf_kind].append(idx)
                if nf_kind == "nf1":
                    child_name = concept_names[str(item[0])]
                    parent_name = concept_names[str(item[1])]
                    neg_samples = sample(concept_list, k=min(10, len(concept_list)))
                    all_train_data.append({"child": child_name, "parent": parent_name,
                                           "negative": neg_samples, "_nf": nf_kind, "_ids": item.tolist()})
                elif nf_kind == "nf2":
                    con1 = concept_names[str(item[0])]
                    con2 = concept_names[str(item[1])]
                    parent_name = concept_names[str(item[2])]
                    child_name = f"{con1} and {con2}"
                    neg_samples = sample(concept_list, k=min(10, len(concept_list)))
                    all_train_data.append({"child": child_name, "parent": parent_name,
                                           "negative": neg_samples, "_nf": nf_kind, "_ids": item.tolist()})
                    conj_data.append({"Concept": child_name, "con1": con1, "con2": con2})
                elif nf_kind == "nf3":
                    child_name = concept_names[str(item[0])]
                    role_name = role_names[str(item[1])]
                    filler_name = concept_names[str(item[2])]
                    parent_name = f"{role_name} some {filler_name}"
                    neg_samples = [f"{role_name} some {n}" for n in sample(concept_list, k=min(10, len(concept_list)))]
                    all_train_data.append({"child": child_name, "parent": parent_name,
                                           "negative": neg_samples, "_nf": nf_kind, "_ids": item.tolist()})
                    exist_data.append({"Concept": parent_name, "role": role_name, "con": filler_name})
                elif nf_kind == "nf4":
                    role_name = role_names[str(item[0])]
                    filler_name = concept_names[str(item[1])]
                    parent_name = concept_names[str(item[2])]
                    child_name = f"{role_name} some {filler_name}"
                    neg_samples = sample(concept_list, k=min(10, len(concept_list)))
                    all_train_data.append({"child": child_name, "parent": parent_name,
                                           "negative": neg_samples, "_nf": nf_kind, "_ids": item.tolist()})
                    exist_data.append({"Concept": child_name, "role": role_name, "con": filler_name})

        random.shuffle(all_train_data)

        # ---- Write train.jsonl (strip internal metadata) ----
        with open(os.path.join(out_dir, "train.jsonl"), "w") as f:
            for d in all_train_data:
                row = {"child": d["child"], "parent": d["parent"], "negative": d["negative"]}
                f.write(json.dumps(row) + "\n")
        with open(os.path.join(out_dir, "train_conj.jsonl"), "w") as f:
            for d in conj_data:
                f.write(json.dumps(d) + "\n")
        with open(os.path.join(out_dir, "train_exist.jsonl"), "w") as f:
            for d in exist_data:
                f.write(json.dumps(d) + "\n")
        logger.info(f"train.jsonl: {len(all_train_data)}, train_exist.jsonl: {len(exist_data)}, "
                     f"train_conj.jsonl: {len(conj_data)}")

        # ---- Sample eval data (10 %, max max_eval) from all axioms ----
        eval_data = self._sample_eval(all_nfs, concept_names, role_names, eval_ratio, max_eval)
        with open(os.path.join(out_dir, "val.json"), "w") as f:
            json.dump(eval_data, f, indent=4)
        total_eval = sum(len(v) for v in eval_data["query_sentences"].values())
        logger.info(f"val.json: {total_eval} eval queries (sampled from train)")

        # ---- Metadata ----
        cn = {str(v[0]): k for k, v in self.concept_names.items()}
        with open(os.path.join(out_dir, "concept_names.json"), "w") as f:
            json.dump(cn, f, indent=2)
        rn = {str(v[0]): k for k, v in self.role_names.items()}
        with open(os.path.join(out_dir, "role_names.json"), "w") as f:
            json.dump(rn, f, indent=2)
        with open(os.path.join(out_dir, "role_inverse.json"), "w") as f:
            json.dump({}, f)
        logger.info(f"concepts: {len(cn)}, roles: {len(rn)}")
        logger.info("OnT data generation complete.")

    # ------------------------------------------------------------------
    def _sample_eval(
        self,
        all_nfs: Dict[str, np.ndarray],
        concept_names: Dict[str, str],
        role_names: Dict[str, str],
        eval_ratio: float,
        max_eval: int,
    ) -> dict:
        """Randomly sample *eval_ratio* of each nf (capped at *max_eval* total)
        and convert to the val.json format used by the evaluator."""
        processed = {"query_sentences": {}, "answer_ids": {}}
        total_axioms = sum(len(v) for v in all_nfs.values())
        budget = min(int(total_axioms * eval_ratio), max_eval)
        # distribute budget proportionally across nf kinds that have data
        nf_counts = {k: len(v) for k, v in all_nfs.items() if len(v) > 0}
        if not nf_counts:
            for k in ["nf1", "nf2", "nf3", "nf4"]:
                processed["query_sentences"][k] = []
                processed["answer_ids"][k] = []
            return processed

        total_with_data = sum(nf_counts.values())
        nf_budgets = {k: max(1, int(budget * c / total_with_data)) for k, c in nf_counts.items()}

        for nf_kind in ["nf1", "nf2", "nf3", "nf4"]:
            data = all_nfs[nf_kind]
            if data.size == 0:
                processed["query_sentences"][nf_kind] = []
                processed["answer_ids"][nf_kind] = []
                continue
            n_sample = min(nf_budgets.get(nf_kind, 0), len(data))
            if n_sample == 0:
                processed["query_sentences"][nf_kind] = []
                processed["answer_ids"][nf_kind] = []
                continue
            indices = np.random.choice(len(data), size=n_sample, replace=False)
            sampled = data[indices]

            sentences, answer_ids, roles, cons = [], [], [], []
            for item in sampled:
                if nf_kind == "nf1":
                    sentences.append(concept_names[str(item[0])])
                    answer_ids.append(int(item[1]))
                elif nf_kind == "nf2":
                    sentences.append(f"{concept_names[str(item[0])]} and {concept_names[str(item[1])]}")
                    answer_ids.append(int(item[2]))
                elif nf_kind == "nf3":
                    rn = role_names[str(item[1])]
                    fn = concept_names[str(item[2])]
                    sentences.append(f"{rn} some {fn}")
                    answer_ids.append(int(item[0]))
                    roles.append(rn)
                    cons.append(fn)
                elif nf_kind == "nf4":
                    rn = role_names[str(item[0])]
                    fn = concept_names[str(item[1])]
                    sentences.append(f"{rn} some {fn}")
                    answer_ids.append(int(item[2]))
                    roles.append(rn)
                    cons.append(fn)

            processed["answer_ids"][nf_kind] = answer_ids
            if nf_kind == "nf1":
                processed["query_sentences"][nf_kind] = [{"name": s} for s in sentences]
            elif nf_kind == "nf2":
                processed["query_sentences"][nf_kind] = [
                    {"name": sentences[idx],
                     "con1": concept_names[str(sampled[idx][0])],
                     "con2": concept_names[str(sampled[idx][1])]}
                    for idx in range(len(sentences))
                ]
            else:
                processed["query_sentences"][nf_kind] = [
                    {"name": sentences[idx], "role": roles[idx], "con": cons[idx]}
                    for idx in range(len(sentences))
                ]

        return processed


def prepare_ontology_data(
    ontology_path: str,
    output_dir: str,
    eval_ratio: float = 0.1,
    max_eval: int = 1000,
) -> str:
    """Prepare OnT training data from an OWL/OFN ontology file.

    All axioms are used for training. A random 10 % subset (capped at
    *max_eval*) is sampled as evaluation data. No test split is produced
    unless a separate ontology is provided via the pipeline.

    Args:
        ontology_path: Path to the OWL/OFN ontology file.
        output_dir: Directory to save the generated training data.
        eval_ratio: Fraction of axioms to sample for evaluation. Default 0.1.
        max_eval: Maximum number of evaluation samples. Default 1000.

    Returns:
        Path to the output directory containing training data.
    """
    os.makedirs(output_dir, exist_ok=True)

    el_data = ELNormalizedData()
    el_data.output_dir = output_dir

    logger.info(f"Loading ontology from {ontology_path}")
    ont, verbalizer = el_data.load(ontology_path)

    logger.info("Extracting role inclusions")
    el_data.get_role_inclusion(ont, verbalizer)

    logger.info("Processing ontology axioms")
    el_data.create_dataset(ont, verbalizer)

    logger.info("Saving OnT training data")
    el_data.save_ont_data(eval_ratio=eval_ratio, max_eval=max_eval)

    return output_dir
