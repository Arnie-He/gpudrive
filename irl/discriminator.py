import os
from typing import Optional, Tuple, Dict, Any, List
from typing_extensions import Annotated
import yaml
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
from box import Box

from gpudrive.integrations.puffer import ppo
from gpudrive.env.env_puffer import PufferGPUDrive

from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.dataset import SceneDataLoader

import pufferlib
import pufferlib.vector
import pufferlib.cleanrl
from rich.console import Console

import typer
from typer import Typer


class Discriminator(nn.Module):
    """
    Discriminator network for GAIL.
    This network distinguishes between expert trajectories and agent trajectories.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        lr: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the discriminator network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions
            lr: Learning rate for the optimizer
            device: Device to run the model on
        """
        super(Discriminator, self).__init__()
        
        self.device = device
        layers = []
        
        # Input layer
        input_dim = state_dim + action_dim
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        
    def forward(self, states, actions):
        """
        Forward pass through the discriminator.
        
        Args:
            states: State batch with shape (batch_size, state_dim)
            actions: Action batch with shape (batch_size, action_dim)
            
        Returns:
            Discriminator outputs with shape (batch_size, 1)
        """
        x = torch.cat([states, actions], dim=1)
        return torch.sigmoid(self.model(x))
    
    
    
    def train_discriminator(
        self, 
        expert_states, 
        expert_actions, 
        policy_states, 
        policy_actions,
        batch_size: int = 64,
        use_gp: bool = True,
        lambda_gp: float = 10.0
    ):
        """
        Train the discriminator using expert and policy data.
        
        Args:
            expert_states: Expert state batch
            expert_actions: Expert action batch
            policy_states: Policy state batch
            policy_actions: Policy action batch
            batch_size: Batch size for training
            use_gp: Whether to use gradient penalty
            lambda_gp: Gradient penalty coefficient
            
        Returns:
            Dictionary containing loss information
        """
        # Move data to device
        expert_states = expert_states.to(self.device)
        expert_actions = expert_actions.to(self.device)
        policy_states = policy_states.to(self.device)
        policy_actions = policy_actions.to(self.device)
        
        # Get discriminator predictions
        expert_preds = self.forward(expert_states, expert_actions)
        policy_preds = self.forward(policy_states, policy_actions)
        
        # Compute loss
        expert_loss = F.binary_cross_entropy(expert_preds, torch.ones_like(expert_preds))
        policy_loss = F.binary_cross_entropy(policy_preds, torch.zeros_like(policy_preds))
        
        gail_loss = expert_loss + policy_loss
        
        
        # Update discriminator
        self.optimizer.zero_grad()
        gail_loss.backward()
        self.optimizer.step()
        
        return {
            "expert_loss": expert_loss.item(),
            "policy_loss": policy_loss.item(),
            "gail_loss": gail_loss.item(),
            "expert_acc": (expert_preds > 0.5).float().mean().item(),
            "policy_acc": (policy_preds < 0.5).float().mean().item()
        }
    
    def get_Q(self, states, actions):
        """
        Calculate reward for policy optimization based on discriminator output.
        
        Args:
            states: State batch
            actions: Action batch
            
        Returns:
            Q batch
        """
        with torch.no_grad():
            d = self.forward(states, actions)
            # Use log(D) as reward to encourage expert-like behavior
            # Different reward formulations are possible:
            # 1. log(D): standard GAIL reward
            # 2. -log(1-D): alternative that may be more stable
            # 3. log(D) - log(1-D): symmetric reward
            return torch.log(d + 1e-8)

