from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import logging
import copy

logger = logging.getLogger(__name__)

class JointHierarchyTransformerTrainer:
    """A trainer that manages multiple trainers and trains them together."""

    def __init__(
        self,
        model: SentenceTransformer,
        trainers: List[Any],
        trainer_weights: Optional[List[float]] = None,
    ):
        """Initialize the joint trainer.
        
        Args:
            model: The shared model used by all trainers
            trainers: List of trainer instances
            trainer_weights: Optional weights for each trainer's loss
        """
        self.model = model
        self.trainers = trainers
        self.trainer_weights = trainer_weights or [1.0/len(trainers)] * len(trainers)
        
        if len(self.trainer_weights) != len(self.trainers):
            raise ValueError("Number of trainer weights must match number of trainers")
            
        # Ensure all trainers share the same model
        for trainer in self.trainers:
            trainer.model = self.model
            
        # Get the first trainer's args as base args
        self.args = copy.deepcopy(trainers[0].args)
        
        # Set up data loaders
        self._setup_dataloaders()

    def _setup_dataloaders(self):
        """Set up data loaders for all trainers."""
        self.train_dataloaders = []
        max_steps = 0
        
        for trainer in self.trainers:
            if hasattr(trainer, 'train_dataloader'):
                dataloader = trainer.train_dataloader
            else:
                # Create dataloader if not exists
                dataloader = DataLoader(
                    trainer.train_dataset,
                    shuffle=True,
                    batch_size=trainer.args.train_batch_size
                )
            self.train_dataloaders.append(dataloader)
            max_steps = max(max_steps, len(dataloader))
            
        # Update total steps
        self.args.num_steps_per_epoch = max_steps
        self.args.num_train_steps = max_steps * self.args.num_train_epochs

    def train(self):
        """Train all trainers together."""
        logger.info(f"Starting joint training with {len(self.trainers)} trainers")
        
        # Training loop
        global_step = 0
        for epoch in range(self.args.num_train_epochs):
            self.model.train()
            
            # Create iterator for each dataloader
            iterators = [iter(dataloader) for dataloader in self.train_dataloaders]
            
            for _ in range(self.args.num_steps_per_epoch):
                total_loss = 0
                
                # Train step for each trainer
                for trainer_idx, (trainer, iterator) in enumerate(zip(self.trainers, iterators)):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        # Restart iterator if exhausted
                        iterator = iter(self.train_dataloaders[trainer_idx])
                        batch = next(iterator)
                        iterators[trainer_idx] = iterator
                    
                    # Compute loss
                    if hasattr(trainer, 'compute_loss'):
                        loss = trainer.compute_loss(self.model, batch)
                    else:
                        # Default loss computation
                        features, labels = batch
                        loss = trainer.loss(features, labels)
                    
                    # Weight the loss
                    weighted_loss = self.trainer_weights[trainer_idx] * loss
                    total_loss += weighted_loss
                
                # Backward pass
                total_loss.backward()
                
                # Optimizer step
                self.trainers[0].optimizer.step()
                self.trainers[0].optimizer.zero_grad()
                
                # Update learning rate
                self.trainers[0].scheduler.step()
                
                global_step += 1
                
                # Evaluation
                if global_step % self.args.evaluation_steps == 0:
                    for trainer in self.trainers:
                        if trainer.evaluator is not None:
                            trainer.evaluator(self.model, output_path=trainer.args.output_path, epoch=epoch, steps=global_step)
            
            # Save checkpoint
            if self.args.output_path is not None and (epoch + 1) % self.args.save_steps == 0:
                self.model.save(self.args.output_path)
                
        logger.info("Joint training completed")
