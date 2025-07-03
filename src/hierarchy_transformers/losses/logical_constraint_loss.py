import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import random

from ..models import HierarchyTransformer

class LogicalConstraintLoss(nn.Module):
    """Loss function that enforces logical constraints in ontology concepts $\exists r. C$ and $C\sqcap D$
    """

    def __init__(
        self,
        model,
        hit_loss,
        data_conj,
        data_exist,
        batch_size,
        conj_weight: float = 1.0,
        exist_weight: float = 1.0, 
        existence_loss_kind: str = 'pair',
        role_inverse: dict[str, str] = None,
    ):
        super().__init__()
        self.model = model
        self.hit_loss = hit_loss

        self.manifold = self.model.hit_model.manifold

        self.conj_weight = conj_weight
        self.exist_weight = exist_weight
        self.batch_size = batch_size

        self.data_conj = data_conj
        self.data_exist = data_exist

        self.role_prompt = "" #'transition factor of relation: ' # add this to the role name
        self.existence_loss_kind = existence_loss_kind
        self.role_inverse = role_inverse

        self.device = next(self.model.parameters()).device
    
    def select_conj(self):
        # extract randomly a batch of sentences from Dataset objects
        batch_indices_conj = random.sample(range(len(self.data_conj)), self.batch_size)
        
        # select the batch and extract the text fields
        batch_conj = self.data_conj.select(batch_indices_conj)
        
        # prepare three separate lists for concept, con1, and con2
        concepts = [item["Concept"] for item in batch_conj]
        con1s = [item["con1"] for item in batch_conj]
        con2s = [item["con2"] for item in batch_conj]
        
        # batch tokenize each list
        sentence_conj = [
            self.model.hit_model.tokenizer(concepts, return_tensors="pt", padding=True, truncation=True).to(self.device),
            self.model.hit_model.tokenizer(con1s, return_tensors="pt", padding=True, truncation=True).to(self.device),
            self.model.hit_model.tokenizer(con2s, return_tensors="pt", padding=True, truncation=True).to(self.device)
        ]

        return sentence_conj

    def select_exist(self):
         # extract randomly a batch of sentences from Dataset objects
        batch_indices_exist = random.sample(range(len(self.data_exist)), self.batch_size)
        
        # select the batch and extract the text fields
        batch_exist = self.data_exist.select(batch_indices_exist)
        
        # prepare three separate lists for concept, con1, and con2
        concepts = [item["Concept"] for item in batch_exist]
        roles = [self.role_prompt + item["role"] for item in batch_exist]
        cons = [item["con"] for item in batch_exist]
        
        # batch tokenize each list
        sentence_exist = [
            self.model.hit_model.tokenizer(concepts, return_tensors="pt", padding=True, truncation=True).to(self.device),
            self.model.hit_model.tokenizer(roles, return_tensors="pt", padding=True, truncation=True).to(self.device),
            self.model.hit_model.tokenizer(cons, return_tensors="pt", padding=True, truncation=True).to(self.device)
        ]
       
        return sentence_exist
    
    def transfer_role_inverse(self):
        # Ensure we iterate through the Dataset in a consistent order
        roles = []
        inverses = []
        for item in self.role_inverse:
            roles.append(item["role"])
            inverses.append(item["inverse"])

        # tokenize each list
        sentence_inverse = [
            self.model.hit_model.tokenizer(roles, return_tensors="pt", padding=True, truncation=True).to(self.device),
            self.model.hit_model.tokenizer(inverses, return_tensors="pt", padding=True, truncation=True).to(self.device)
        ]

        return sentence_inverse

    def conj_loss(self, sentence_conjs, neg_samples):
        # compute losses for conjunctions
        reps_conj = [self.model(sentence_conj)["sentence_embedding"] for sentence_conj in sentence_conjs]
        conj_emb, conj1_emb, conj2_emb = reps_conj

        neg_samples_1 = neg_samples[torch.randperm(neg_samples.size(0))]
        neg_samples_2 = neg_samples[torch.randperm(neg_samples.size(0))]

        conj_loss =  (
            self.hit_loss.from_tensor(conj_emb, conj1_emb, neg_samples_1)["loss"]
            + self.hit_loss.from_tensor(conj_emb, conj2_emb, neg_samples_2)["loss"]
            )/2

        return conj_loss

    def exist_loss(self, sentence_exists, neg_samples):
        # compute losses for existence
        if self.existence_loss_kind == 'hit':
            exist_emb = self.model.forward_existence(sentence_exists)
            sentence_emb, composed_emb = exist_emb["sentence_embedding"], exist_emb["rotated_embedding"]

            # shuffle neg_samples
            neg_samples_1 = neg_samples[torch.randperm(neg_samples.size(0))]    
            neg_samples_2 = neg_samples[torch.randperm(neg_samples.size(0))]

            # exist_loss = self.manifold.dist(exist_sentence_emb, exists_rotate_emb).mean()
            exist_loss = (
                self.hit_loss.from_tensor(sentence_emb, composed_emb, neg_samples_1)["loss"]
                + self.hit_loss.from_tensor(composed_emb, sentence_emb,  neg_samples_2)["loss"]
                )/2
        elif self.existence_loss_kind == 'pair':
            exist_loss = self.model.pair_loss_existence(sentence_exists)
        elif self.existence_loss_kind == 'dist':
            exist_emb = self.model.forward_existence(sentence_exists)
            exist_sentence_emb, exist_rotate_emb = exist_emb["sentence_embedding"], exist_emb["rotated_embedding"]

            exist_loss = self.manifold.dist(exist_sentence_emb, exist_rotate_emb).mean()
            # exist_loss = torch.norm(exist_sentence_emb - exist_rotate_emb, dim=-1).mean()
        else:
            raise ValueError(f"Unknown existence_loss_kind: {self.existence_loss_kind}")
        
        return exist_loss
    

    def forward(
        self, 
        sentence_features,
        labels: torch.Tensor
     ) -> Dict[str, torch.Tensor]:
        """Compute the logical constraint loss.
        Returns:
            Dictionary containing the loss components and total loss.
        """
        neg_samples = self.model(sentence_features[2])['sentence_embedding']
        self.batch_size = neg_samples.size(0)

        sentence_exists = self.select_exist()

        # regard the embedding of con in sentence_exists as negative samples
        # neg_samples = self.model(sentence_exists[2])['sentence_embedding']

        exist_loss = self.exist_loss(sentence_exists, neg_samples)   

        if self.data_conj:
            sentence_conjs = self.select_conj()
            conj_loss = self.conj_loss(sentence_conjs, neg_samples)
        else:
            conj_loss = torch.tensor(0.0, device=self.device)

        hit_loss = self.hit_loss(sentence_features, labels)

        total_loss = self.conj_weight * conj_loss + self.exist_weight * exist_loss + hit_loss['loss']
        
        return {
            "loss": total_loss,
            "conj_loss": conj_loss,
            "exist_loss": exist_loss,
            "cluster_loss": hit_loss["cluster_loss"],
            "centri_loss": hit_loss["centri_loss"],
        }

    def get_config_dict(self):
        return {
            "conj_weight": self.conj_weight,
            "exist_weight": self.exist_weight,
            "cluster_weight": self.hit_loss.cluster_weight,
            "centri_weight": self.hit_loss.centri_weight,
        }
