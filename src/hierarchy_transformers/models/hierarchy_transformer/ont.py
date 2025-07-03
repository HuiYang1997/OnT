import torch
import torch.nn as nn
from typing import Dict, Any
from .hit import HierarchyTransformer

class OntologyTransformer(nn.Module):
    """A model that extends HierarchyTransformer to ontology embeddings with role embedding and role model."""
    
    def __init__(self, base_model: HierarchyTransformer, role_emd_mode, role_model_mode):
        super().__init__()
        self.hit_model = base_model
        self.dim = self.hit_model.embed_dim
        self.role_emd_mode = role_emd_mode
        self.role_model_mode = role_model_mode

        assert self.dim % 2 == 0, "Embedding dimension must be even"

        # Define role_model as a two-layer neural network for role transformation
        if self.role_model_mode == 'rotation':
            output_dim = self.dim//2
        elif self.role_model_mode == 'transition':
            output_dim = self.dim
        else:
            raise ValueError(f"Unknown role_model_mode: {self.role_model_mode}")

        self.role_model = nn.Linear(self.dim, 1+output_dim)
        self.margin = 0.1
        
    def forward(self, features) -> Dict[str, torch.Tensor]:
        """Forward pass that maintains compatibility with HierarchyTransformer interface."""
        return self.hit_model(features)

    def get_role_embedding(self, role_feature):
        if self.role_emd_mode == 'tokenEmbedding':
            # Use the embeddings from hit_model directly and apply pooling
            token_embeddings = self.hit_model._first_module().auto_model.embeddings.word_embeddings(role_feature['input_ids'])
            attention_mask = role_feature['attention_mask'].unsqueeze(-1)  # [batch_size, seq_len, 1]
            # Mean pooling with attention mask
            role_emb = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            role_emb = self.role_model(role_emb)
        elif self.role_emd_mode == 'sentenceEmbedding':
            # role_emd = self.hit_model(features[1])['sentence_embedding'] 
            role_emb = self.role_model(
                self.hit_model(role_feature)['sentence_embedding']
                )
        else:
            raise ValueError(f"Unknown role_emd_mode: {self.role_emd_mode}")
        
        
        if self.role_model_mode == 'rotation':
            # Normalize rotation angles to [-π, π] for better stability
            rotation = torch.tanh(role_emb[..., 1:]) * torch.pi
        else:
            rotation = role_emb[..., 1:]
        
        scaling_emb = torch.exp(role_emb[..., 0])
        
        return scaling_emb, rotation
    
    def pair_loss_existence(self, features):
        complex_emd = self.hit_model(features[0])['sentence_embedding']
        con_emd = self.hit_model(features[2])['sentence_embedding']

        # compute the distance between con_emb[:-1] and con_emb[1:]
        slide_dist_con = self.hit_model.manifold.dist(con_emd[:-1], con_emd[1:])
        slide_dist_complex = self.hit_model.manifold.dist(complex_emd[:-1], complex_emd[1:])

        # define a loss that encourage the distance of slide_dist_con to be less than the distance of slide_dist_complex
        loss_dist_hyp = torch.relu(slide_dist_complex - slide_dist_con + self.margin)

        complex_hyper_norms = self.hit_model.manifold.dist0(complex_emd)
        con_hyper_norms = self.hit_model.manifold.dist0(con_emd)

        # repeat the above progress but replace the manifold distance by the differen of norm over the manifold
        slid_norm_con = con_hyper_norms[:-1] - con_hyper_norms[1:]
        slid_norm_complex = complex_hyper_norms[:-1] - complex_hyper_norms[1:]

        # define a loss that encourage the distance of slide_dist_con to be less than the distance of slide_dist_complex
        loss_dist_norm = torch.relu(slid_norm_complex - slid_norm_con + self.margin)

        return (loss_dist_hyp + loss_dist_norm).mean()
    
    def existence_emb(self, features) -> Dict[str, torch.Tensor]:
        con_emd = self.hit_model(features[1])['sentence_embedding']
        scaling, rotation = self.get_role_embedding(features[0])
        
        if self.role_model_mode == 'rotation':
            # Split concept embedding into two parts for rotation
            con_emb1, con_emb2 = torch.chunk(con_emd, 2, dim=-1)
            
            # Compute rotation using stable sine/cosine
            sin_theta = torch.sin(rotation)
            cos_theta = torch.cos(rotation)
        
            # Apply rotation using the 2D rotation matrix formula
            rotated_emb1 = con_emb1 * cos_theta - con_emb2 * sin_theta
            rotated_emb2 = con_emb1 * sin_theta + con_emb2 * cos_theta
        
            # Combine rotated embeddings and scale in hyperbolic space
            rotated_emb = torch.cat([rotated_emb1, rotated_emb2], dim=-1)
        elif self.role_model_mode == 'transition':
            # realize the rotation by transition on the manifold
            rotated_emb = self.hit_model.manifold.expmap(
                con_emd,
                rotation
            )
        else:
            raise ValueError(f"Unknown role_model_mode: {self.role_model_mode}")
        
        # Use sigmoid for scaling to ensure positive values and stable training
        scaled_emb = self.hit_model.manifold.mobius_scalar_mul(
            scaling.unsqueeze(-1),
            rotated_emb
        )
        
        return rotated_emb


    def forward_existence(self, features) -> Dict[str, torch.Tensor]:
        # output the embedding of concept of the form $\exist r. C$ by rotation and text description
        concept_emd = self.hit_model(features[0])['sentence_embedding']
        scaled_emb = self.existence_emb(features[1:])
        
        return {"sentence_embedding": concept_emd, "rotated_embedding": scaled_emb}
    
    def score_hierarchy(self, child_embeds: torch.Tensor, parent_embeds: torch.Tensor, weight: float=0.0) -> torch.Tensor:
        """Maintain compatibility with HierarchyTransformer's scoring function."""
        distances = self.manifold.dist(child_embeds, parent_embeds)
        child_norms = self.manifold.dist0(child_embeds)
        parent_norms = self.manifold.dist0(parent_embeds)
        dist_norm = distances + weight * (parent_norms - child_norms)
        return -dist_norm
    
    @property
    def manifold(self):
        """Access to the base model's manifold."""
        return self.hit_model.manifold
    
    @property
    def embed_dim(self):
        """Access to the base model's embedding dimension."""
        return self.hit_model.embed_dim
    
    def get_sentence_embedding_dimension(self) -> int:
        """Required by SentenceTransformer interface."""
        return self.hit_model.get_sentence_embedding_dimension()
    
    def tokenize(self, *args, **kwargs):
        """Required by SentenceTransformer interface."""
        return self.hit_model.tokenize(*args, **kwargs)
    
    def get_config_dict(self):
        """Required by SentenceTransformer interface."""
        return self.hit_model.get_config_dict()
    
    def save(self, output_path: str, *args, **kwargs):
        """Save the model with all its attributes.
        
        Args:
            output_path: Path where the model will be saved (required as first argument)
        """
        if output_path is None or not isinstance(output_path, str):
            raise ValueError("'output_path' must be provided as the first argument and must be a string")
            
        # First save the base HierarchyTransformer model
        self.hit_model.save(output_path, *args, **kwargs)
        
        # Save additional wrapper attributes
        import json
        import os
        wrapper_config = {
            'role_emd_mode': self.role_emd_mode,
            'role_model_mode': self.role_model_mode
        }
        
        # Save wrapper configuration to a separate JSON file
        with open(os.path.join(output_path, 'wrapper_config.json'), 'w') as f:
            json.dump(wrapper_config, f)
            
        # Save role_model state dict
        torch.save(self.role_model.state_dict(), os.path.join(output_path, 'role_model.pt'))
        
        return output_path
    
    @staticmethod
    def load(input_path: str):
        """Load the model with all its attributes.
        
        Args:
            input_path: Path from which the model will be loaded
            
        Returns:
            Loaded HierarchyTransformerWrapper model
        """
        # Load base HierarchyTransformer model
        base_model = HierarchyTransformer.from_pretrained(input_path)
        
        # Load wrapper configuration
        import json
        import os
        wrapper_config_path = os.path.join(input_path, 'wrapper_config.json')
        
        # Check if wrapper config exists
        if os.path.exists(wrapper_config_path):
            with open(wrapper_config_path, 'r') as f:
                wrapper_config = json.load(f)
            
            role_emd_mode = wrapper_config.get('role_emd_mode', 'sentenceEmbedding')
            role_model_mode = wrapper_config.get('role_model_mode', 'rotation')
        else:
            # Use default values if config not found
            role_emd_mode = 'sentenceEmbedding'
            role_model_mode = 'rotation'
            print(f"Warning: wrapper_config.json not found in {input_path}, using default values.")
        
        # Create wrapper instance with loaded configuration
        wrapper = HierarchyTransformerWrapper(base_model, role_emd_mode, role_model_mode)
        
        # Load role_model weights if they exist
        role_model_path = os.path.join(input_path, 'role_model.pt')
        if os.path.exists(role_model_path):
            wrapper.role_model.load_state_dict(torch.load(role_model_path))
        
        return wrapper
