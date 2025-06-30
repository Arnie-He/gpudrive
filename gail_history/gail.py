"""
This implementation is adapted from the demo in PufferLib by Joseph Suarez,
which in turn is adapted from Costa Huang's CleanRL PPO + LSTM implementation.
Links
- PufferLib: https://github.com/PufferAI/PufferLib/blob/dev/demo.py
- Cleanrl: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
"""

import os
from typing import Optional
from typing_extensions import Annotated
import yaml
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from box import Box
from collections import deque
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

from gpudrive.integrations.puffer import ppo
from gpudrive.env.env_puffer import PufferGPUDrive
from gpudrive.visualize.utils import img_from_fig

from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.dataset import SceneDataLoader
from baselines.imitation_data_generation import generate_state_action_pairs
from irl.storage import save_trajectory

import pufferlib
import pufferlib.vector
import pufferlib.cleanrl
from rich.console import Console

import typer
from typer import Typer

app = Typer()

def get_model_parameters(policy):
    """Helper function to count the number of trainable parameters."""
    params = filter(lambda p: p.requires_grad, policy.parameters())
    return sum([np.prod(p.size()) for p in params])


class Discriminator(nn.Module):
    """Discriminator network for GAIL that distinguishes expert from policy trajectories based on states only."""
    
    def __init__(self, obs_dim, hidden_dim=256, dropout=0.0):
        super().__init__()
        
        # Store dimensions for later use
        self.obs_dim = obs_dim
        
        # Input is state only
        input_dim = obs_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, states):
        """
        Forward pass of discriminator.
        
        Args:
            states: Tensor of shape (batch_size, obs_dim)
            
        Returns:
            logits: Tensor of shape (batch_size, 1) - logit for real (expert) vs fake (policy)
        """
        # Ensure states have correct shape
        if states.dim() > 2:
            states = states.reshape(states.shape[0], -1)
            
        return self.network(states)
    
    def predict_reward(self, states, reward_type="scaled_log"):
        """
        Predict reward based on discriminator output.
        Multiple reward formulations available to prevent policy collapse.
        
        Args:
            states: State observations
            reward_type: Type of reward calculation
                - "scaled_log": Scaled -log(1-D) (default, most stable)
                - "original": Original -log(1-D) 
                - "wgan": WGAN-style D(s)
                - "least_squares": Least squares discriminator reward
        """
        with torch.no_grad():
            logits = self.forward(states)
            # Convert logits to probability of being expert
            prob_expert = torch.sigmoid(logits)
            
            # Clamp probabilities to prevent numerical issues
            prob_expert = torch.clamp(prob_expert, 1e-7, 1 - 1e-7)
            
            if reward_type == "scaled_log":
                # RECOMMENDED: Scaled version of GAIL reward for stability
                reward = -torch.log(1 - prob_expert)
                reward = torch.clamp(reward, -10.0, 10.0)  # Clamp extreme rewards
                reward = reward * 0.1  # Scale down rewards
                
            elif reward_type == "original":
                # Original GAIL reward: -log(1 - D(s))
                reward = -torch.log(1 - prob_expert)
                
            elif reward_type == "wgan":
                # WGAN-style reward: D(s) directly
                # More stable but may not work as well for imitation
                reward = prob_expert
                
            elif reward_type == "least_squares":
                # Least squares discriminator reward
                # Often more stable than log-based rewards
                reward = -(1 - prob_expert) ** 2
                reward = reward * 0.5  # Scale appropriately
                
            else:
                raise ValueError(f"Unknown reward_type: {reward_type}")
            
            return reward.squeeze(-1)

    def gradient_penalty(self, real_samples, fake_samples, device, lambda_gp=10.0):
        """
        Compute WGAN-GP style gradient penalty for improved stability.
        
        Args:
            real_samples: Real state samples
            fake_samples: Generated/policy state samples  
            device: Device to run computation on
            lambda_gp: Gradient penalty coefficient
            
        Returns:
            gradient_penalty: Computed gradient penalty loss
        """
        # Ensure both samples are on the correct device
        real_samples = real_samples.to(device)
        fake_samples = fake_samples.to(device)
        
        batch_size = real_samples.shape[0]
        
        # Generate random interpolation factors
        alpha = torch.rand(batch_size, 1, device=device)
        # Expand alpha to match sample dimensions
        alpha = alpha.expand_as(real_samples)
        
        # Create interpolated samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Forward pass on interpolated samples
        d_interpolated = self.forward(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty


class ExpertDataset:
    """Dataset for storing and sampling expert trajectories."""
    def __init__(self, expert_obs, device='cpu'):
        self.expert_obs = expert_obs
        self.device = device
        self.size = len(expert_obs)
        
    def sample(self, batch_size):
        """Sample a batch of expert states."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.expert_obs[indices]


class GAILTrainer:
    def __init__(self, config, discriminator, expert_dataset, policy_trainer_data, env_config=None):
        self.config = config
        self.env_config = env_config  # Store environment config for action_type access
        self.discriminator = discriminator
        self.expert_dataset = expert_dataset
        self.policy_trainer_data = policy_trainer_data
        
        # Discriminator optimizer
        self.disc_optimizer = torch.optim.Adam(
            discriminator.parameters(), 
            lr=config.discriminator_lr,
            betas=(0.5, 0.999),
            weight_decay=getattr(config, 'discriminator_weight_decay', 0.0)
        )
        
        # Buffers for policy trajectories
        self.policy_obs_buffer = deque(maxlen=config.policy_buffer_size)
        
        # Logging
        self.disc_losses = []
        self.disc_expert_acc = []
        self.disc_policy_acc = []
        self.grad_penalty_losses = []
        
        self.use_gradient_penalty = config.use_gradient_penalty
        self.gradient_penalty_lambda = config.gradient_penalty_lambda
        self.disc_updates_per_policy_update = config.disc_updates_per_policy_update
        self.update_counter = 0
        
    def add_policy_data(self, obs):
        """Add policy-generated trajectories to buffer."""
        self.policy_obs_buffer.extend(obs.cpu())
    
    def train_discriminator(self, num_epochs=5):
        """
        Train discriminator to distinguish expert from policy data.
        Uses balanced sampling and gradient penalty for stability.
        """
        if len(self.policy_obs_buffer) < self.config.min_policy_data:
            print(f"Skipping discriminator training - insufficient policy data ({len(self.policy_obs_buffer)} < {self.config.min_policy_data})")
            return
            
        # MODIFICATION 1: Use balanced sampling - always sample same number from expert as policy
        batch_size = len(self.policy_obs_buffer)
        
        disc_losses = []
        expert_accs = []
        policy_accs = []
        grad_penalty_losses = []
        
        for epoch in range(num_epochs):
            # Sample equal amounts of expert and policy data
            expert_obs = self.expert_dataset.sample(batch_size)
            
            # Sample policy data
            policy_indices = random.sample(range(len(self.policy_obs_buffer)), batch_size)
            policy_obs = torch.stack([self.policy_obs_buffer[i] for i in policy_indices]).to(self.config.device)
            
            # Create labels: 1 for expert, 0 for policy
            expert_labels = torch.ones(batch_size, 1, device=self.config.device)
            policy_labels = torch.zeros(batch_size, 1, device=self.config.device)
            
            # Add label smoothing to prevent discriminator overfitting
            label_smoothing = 0.1  # Smooth labels by 10%
            expert_labels = expert_labels * (1 - label_smoothing) + 0.5 * label_smoothing
            policy_labels = policy_labels * (1 - label_smoothing) + 0.5 * label_smoothing
            
            # Combine data
            all_obs = torch.cat([expert_obs, policy_obs])
            all_labels = torch.cat([expert_labels, policy_labels])
            
            # Forward pass
            self.disc_optimizer.zero_grad()
            logits = self.discriminator(all_obs)
            
            # Binary cross entropy loss
            bce_loss = F.binary_cross_entropy_with_logits(logits, all_labels)
            
            # MODIFICATION 2: Add gradient penalty for stability (WGAN-GP style)
            total_loss = bce_loss
            grad_penalty = 0.0
            
            if self.use_gradient_penalty:
                # Prepare samples for gradient penalty
                expert_state = expert_obs.reshape(expert_obs.shape[0], -1)
                policy_state = policy_obs.reshape(policy_obs.shape[0], -1)
                
                # Ensure both tensors are on the same device before gradient penalty
                expert_state = expert_state.to(self.config.device)
                policy_state = policy_state.to(self.config.device)
                
                grad_penalty = self.discriminator.gradient_penalty(
                    expert_state, 
                    policy_state, 
                    self.config.device, 
                    self.gradient_penalty_lambda
                )
                total_loss = bce_loss + grad_penalty
                grad_penalty_losses.append(grad_penalty.item())
            
            # Backward pass
            total_loss.backward()
            
            # Optional: Gradient clipping for additional stability
            if hasattr(self.config, 'grad_clip_norm') and self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.grad_clip_norm)
            
            self.disc_optimizer.step()
            
            # Track metrics
            disc_losses.append(total_loss.item())
            
            # Calculate accuracies
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                expert_preds = probs[:batch_size] > 0.5
                policy_preds = probs[batch_size:] < 0.5
                
                expert_acc = expert_preds.float().mean().item()
                policy_acc = policy_preds.float().mean().item()
                
                expert_accs.append(expert_acc)
                policy_accs.append(policy_acc)
                
            # Early stopping if discriminator becomes too perfect
            # This prevents the discriminator from overfitting and causing policy collapse
            avg_acc = (expert_acc + policy_acc) / 2
            if avg_acc > 0.95:  # If discriminator is >95% accurate, stop training early
                print(f"Early stopping discriminator training at epoch {epoch+1} due to high accuracy: {avg_acc:.3f}")
                break
        
        
        # Store metrics
        if disc_losses:
            self.disc_losses.extend(disc_losses)
            self.disc_expert_acc.extend(expert_accs)
            self.disc_policy_acc.extend(policy_accs)
            if grad_penalty_losses:
                self.grad_penalty_losses.extend(grad_penalty_losses)
            
            # Print training summary
            avg_loss = sum(disc_losses) / len(disc_losses)
            avg_expert_acc = sum(expert_accs) / len(expert_accs)
            avg_policy_acc = sum(policy_accs) / len(policy_accs)
            avg_total_acc = (avg_expert_acc + avg_policy_acc) / 2
            avg_grad_penalty = sum(grad_penalty_losses) / len(grad_penalty_losses)

            # Log to wandb
        if hasattr(self.policy_trainer_data, 'wandb') and self.policy_trainer_data.wandb:
            self.policy_trainer_data.wandb.log({
                'discriminator/loss': avg_loss,
                'discriminator/expert_accuracy': avg_expert_acc,
                'discriminator/policy_accuracy': avg_policy_acc,
                # 'discriminator/gradient_penalty': avg_grad_penalty,
                # 'discriminator/batch_size': batch_size,
                'global_step': self.policy_trainer_data.global_step
            })
            
            # print(f"Discriminator Training - Loss: {avg_loss:.4f}, Expert Acc: {avg_expert_acc:.3f}, Policy Acc: {avg_policy_acc:.3f}, Total Acc: {avg_total_acc:.3f}")
            
            # MODIFICATION: Adaptive discriminator learning rate based on accuracy
            # Reduce learning rate if discriminator is becoming too accurate
            if avg_total_acc > 0.85:
                for param_group in self.disc_optimizer.param_groups:
                    param_group['lr'] *= 0.85  # Reduce learning rate by 15%
                print(f"Reduced discriminator learning rate to: {param_group['lr']:.6f}")
            elif avg_total_acc < 0.5:
                for param_group in self.disc_optimizer.param_groups:
                    param_group['lr'] *= 1.05  # Increase learning rate by 5%
                    param_group['lr'] = min(param_group['lr'], self.config.discriminator_lr)  # Cap at original LR
                print(f"Increased discriminator learning rate to: {param_group['lr']:.6f}")

    def should_update_discriminator(self):
        """
        Determine if discriminator should be updated based on adaptive criteria.
        This prevents the discriminator from getting too far ahead of the policy.
        """
        # Basic update frequency check
        if self.update_counter % self.disc_updates_per_policy_update != 0:
            return False
            
        # Check if we have enough policy data
        if len(self.policy_obs_buffer) < self.config.min_policy_data:
            return False
            
        # MODIFICATION: Adaptive update schedule based on discriminator performance
        # If discriminator is too accurate, skip updates to let policy catch up
        if len(self.disc_expert_acc) > 10 and len(self.disc_policy_acc) > 10:
            recent_expert_acc = sum(self.disc_expert_acc[-10:]) / 10
            recent_policy_acc = sum(self.disc_policy_acc[-10:]) / 10
            recent_total_acc = (recent_expert_acc + recent_policy_acc) / 2
            
            # If discriminator is too accurate (>90%), reduce update frequency
            if recent_total_acc > 0.9:
                return self.update_counter % (self.disc_updates_per_policy_update * 3) == 0
            # If discriminator is struggling (<60%), increase update frequency  
            elif recent_total_acc < 0.6:
                return True
                
        return True

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return pufferlib.namespace(**config)


def load_expert_data(config, vecenv):
    """Load expert demonstrations for GAIL training."""
    
    save_path = f"irl/data/puffer_{config.train.seed}"
    trajectory_file = f"{save_path}/trajectory_0.npz"
    global_file = f"{save_path}/global/global_trajectory_0.npz"
    
    # Check if we should remake data or if data doesn't exist
    remake_data = getattr(config, 'expertdata', {}).get('remake', False)
    data_exists = os.path.exists(trajectory_file) and os.path.exists(global_file)
    
    # Determine action space type from environment
    action_space_type = getattr(config.environment, 'action_type', 'discrete')
    print(f"Expert data generation using {action_space_type} actions")
    
    if remake_data or not data_exists:
        if remake_data:
            print("Remaking expert demonstrations (config.expertdata.remake=True)...")
        else:
            print("Expert data not found. Generating expert demonstrations...")
            
        save_trajectory(
            env=vecenv.env,
            save_path=save_path,
            save_index=0,
            action_space_type=action_space_type,
            use_action_indices=True if action_space_type == "discrete" else False,
            save_visualization=False,
            render_index=[0, 2]
        )
    else:
        print(f"Loading existing expert data from {save_path}...")
    
    # Load expert data - only observations needed
    expert_data = np.load(trajectory_file)
    expert_obs = expert_data["obs"]
    
    # Load expert global data(not used)
    expert_global_data = np.load(global_file)
    expert_global_pos = expert_global_data["ego_global_pos"]

    collision = expert_data["veh_collision"]
    off_road = expert_data["off_road"]
    goal_achieved = expert_data["goal_achieved"]

    print(f"Off-road rate: {off_road:.3f}, Vehicle collision rate: {collision:.3f}, Goal rate: {goal_achieved:.3f}, using non-collided trajectories")

    return expert_obs


def make_agent(env, config):
    """Create a policy based on the environment."""

    if config.continue_training:
        print("Loading checkpoint...")
        # Load checkpoint
        saved_cpt = torch.load(
            f=config.model_cpt,
            map_location=config.train.device,
            weights_only=False,
        )
        policy = NeuralNet(
            input_dim=saved_cpt["model_arch"]["input_dim"],
            action_dim=saved_cpt["action_dim"],
            hidden_dim=saved_cpt["model_arch"]["hidden_dim"],
            config=config.environment,
        )

        # Load the model parameters
        policy.load_state_dict(saved_cpt["parameters"])

        return policy

    else:
        # Start from scratch
        # Detect if action space is continuous or discrete
        if hasattr(env.single_action_space, 'n'):
            # Discrete action space
            action_dim = env.single_action_space.n
            continuous_actions = False
            print(f"Policy: Using DISCRETE actions with {action_dim} possible actions")
        else:
            # Continuous action space
            action_dim = env.single_action_space.shape[0]
            continuous_actions = True
            print(f"Policy: Using CONTINUOUS actions with {action_dim}D action space")
            print(f"Action bounds: {env.single_action_space.low} to {env.single_action_space.high}")
            
        return NeuralNet(
            input_dim=config.train.network.input_dim,
            action_dim=action_dim,
            hidden_dim=config.train.network.hidden_dim,
            config=config.environment,
            continuous_actions=continuous_actions,
            using_shared_embedding=config.train.network.using_shared_embedding,
            action_low=env.single_action_space.low if continuous_actions else None,
            action_high=env.single_action_space.high if continuous_actions else None,
        )


def train_gail(args, vecenv):
    """Main GAIL training loop."""
    # CUDA context fix: Initialize CUDA early if using GPU
    if torch.cuda.is_available() and args.train.device == "cuda":
        torch.cuda.init()  # Initialize CUDA context early
        torch.cuda.empty_cache()  # Clear any lingering GPU memory
    
    # Create policy (generator)
    policy = make_agent(env=vecenv.driver_env, config=args).to(args.train.device)
    args.train.network.num_parameters = get_model_parameters(policy)
    args.train.env = args.environment.name

    # Load expert data
    expert_obs = load_expert_data(args, vecenv)
    expert_obs = torch.tensor(expert_obs).to(args.train.device).reshape(-1, expert_obs.shape[-1])
    expert_dataset = ExpertDataset(expert_obs, device=args.train.device)
    
    print(f"expert_obs shape: {expert_obs.shape}")

    # Create discriminator
    obs_dim = expert_obs.shape[-1]
    
    print(f"Discriminator setup: obs_dim={obs_dim}, using state-only discriminator")
    
    discriminator = Discriminator(
        obs_dim=obs_dim,
        hidden_dim=args.train.discriminator_hidden_dim,
        dropout=args.train.discriminator_dropout
    ).to(args.train.device)
    
    # Create PPO trainer
    args.wandb = init_wandb(args, args.train.exp_id, id=args.train.exp_id)
    args.train.__dict__.update(dict(args.wandb.config.train))
    ppo_data = ppo.create(args.train, vecenv, policy, wandb=args.wandb)
    
    # Override PPO optimizer with L2 regularization for policy
    policy_weight_decay = getattr(args.train, 'policy_weight_decay', 0.0)
    value_weight_decay = getattr(args.train, 'value_weight_decay', 0.0)
    
    if policy_weight_decay > 0 or value_weight_decay > 0:
        # Separate parameters for different weight decay values
        if hasattr(policy, 'actor') and hasattr(policy, 'critic'):
            # If policy has separate actor/critic
            policy_params = list(policy.actor.parameters())
            value_params = list(policy.critic.parameters())
        else:
            # For networks where actor/critic share parameters, apply policy weight decay
            policy_params = list(policy.parameters())
            value_params = []
        
        # Create parameter groups with different weight decay
        param_groups = []
        if policy_params:
            param_groups.append({
                'params': policy_params,
                'weight_decay': policy_weight_decay,
                'lr': float(args.train.learning_rate)
            })
        if value_params:
            param_groups.append({
                'params': value_params, 
                'weight_decay': value_weight_decay,
                'lr': float(args.train.learning_rate)
            })
        
        if param_groups:
            ppo_data.optimizer = torch.optim.Adam(param_groups, eps=1e-5)
            print(f"Applied L2 regularization - Policy: {policy_weight_decay}, Value: {value_weight_decay}")
        else:
            # Fallback: apply policy weight decay to all parameters
            ppo_data.optimizer = torch.optim.Adam(
                policy.parameters(), 
                lr=float(args.train.learning_rate), 
                eps=1e-5, 
                weight_decay=policy_weight_decay
            )
            print(f"Applied L2 regularization to all parameters: {policy_weight_decay}")
    
    # Create GAIL trainer
    gail_trainer = GAILTrainer(args.train, discriminator, expert_dataset, ppo_data, args.environment)
    
    step_count = 0
    discriminator_train_count = 0
    while ppo_data.global_step < args.train.total_timesteps:
        try:
            # Standard PPO rollout but collect data for discriminator
            ppo.evaluate(ppo_data)
            
            # Extract rollout data for discriminator training
            obs_data = ppo_data.experience.obs[:ppo_data.experience.ptr]
            action_data = ppo_data.experience.actions[:ppo_data.experience.ptr]
            
            # Add policy data to GAIL trainer
            gail_trainer.add_policy_data(obs_data)
            
            # Replace environment rewards with discriminator rewards
            with torch.no_grad():
                disc_rewards = discriminator.predict_reward(
                    obs_data.to(args.train.device),
                    reward_type="scaled_log"
                )
                # Reshape to match experience buffer format
                disc_rewards = disc_rewards.view(ppo_data.experience.rewards[:ppo_data.experience.ptr].shape)
                # print(f"disc_rewards samples: {disc_rewards[:10]}")
                # print(f"real rewards samples: {ppo_data.experience.rewards[:ppo_data.experience.ptr][:10]}")
                ppo_data.experience.rewards[:ppo_data.experience.ptr] = disc_rewards.cpu()
                # print(f"real rewards samples after: {ppo_data.experience.rewards[:ppo_data.experience.ptr][:10]}")
            
            if gail_trainer.should_update_discriminator():
                gail_trainer.train_discriminator(args.train.discriminator_epochs)
                discriminator_train_count += 1
                
                # Log discriminator training frequency
                if hasattr(ppo_data, 'wandb') and ppo_data.wandb:
                    ppo_data.wandb.log({
                        'discriminator/train_frequency': discriminator_train_count / (step_count + 1),
                        'discriminator/total_updates': discriminator_train_count,
                        'global_step': ppo_data.global_step
                    })
            
            # Train policy with discriminator rewards
            ppo.train(ppo_data)
            
            # Update the counter after checking
            gail_trainer.update_counter += 1
            
        except KeyboardInterrupt:
            ppo.close(ppo_data)
            os._exit(0)
        except Exception as e:
            print(f"An error occurred: {e}")
            Console().print_exception()
            os._exit(1)

    ppo.evaluate(ppo_data)
    ppo.close(ppo_data)


def init_wandb(args, name, id=None, resume=True):
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args.wandb.project,
        entity=args.wandb.entity,
        group=args.wandb.group,
        mode=args.wandb.mode,
        tags=args.wandb.tags,
        config={
            "environment": dict(args.environment),
            "train": dict(args.train),
            "vec": dict(args.vec),
        },
        name=name,
        save_code=True,
        resume=False,
    )

    return wandb


# def sweep(args, project="PPO", sweep_name="my_sweep"):
#     """Initialize a WandB sweep with hyperparameters."""
#     sweep_id = wandb.sweep(
#         sweep=dict(
#             method="random",
#             name=sweep_name,
#             metric={"goal": "maximize", "name": "environment/episode_return"},
#             parameters={
#                 "learning_rate": {
#                     "distribution": "log_uniform_values",
#                     "min": 1e-4,
#                     "max": 1e-1,
#                 },
#                 "batch_size": {"values": [512, 1024, 2048]},
#                 "minibatch_size": {"values": [128, 256, 512]},
#             },
#         ),
#         project=project,
#     )
#     wandb.agent(sweep_id, lambda: train(args), count=100)


@app.command()
def run(
    config_path: Annotated[
        str, typer.Argument(help="The path to the configuration file")
    ] = "irl/config/gail_config_fast.yaml",
    *,
    # fmt: off
    # Environment options
    num_worlds: Annotated[Optional[int], typer.Option(help="Number of parallel envs")] = None,
    k_unique_scenes: Annotated[Optional[int], typer.Option(help="The number of unique scenes to sample")] = None,
    collision_weight: Annotated[Optional[float], typer.Option(help="The weight for collision penalty")] = None,
    off_road_weight: Annotated[Optional[float], typer.Option(help="The weight for off-road penalty")] = None,
    goal_achieved_weight: Annotated[Optional[float], typer.Option(help="The weight for goal-achieved reward")] = None,
    dist_to_goal_threshold: Annotated[Optional[float], typer.Option(help="The distance threshold for goal-achieved")] = None,
    sampling_seed: Annotated[Optional[int], typer.Option(help="The seed for sampling scenes")] = None,
    obs_radius: Annotated[Optional[float], typer.Option(help="The radius for the observation")] = None,
    collision_behavior: Annotated[Optional[str], typer.Option(help="The collision behavior; 'ignore' or 'remove'")] = None,
    remove_non_vehicles: Annotated[Optional[int], typer.Option(help="Remove non-vehicles from the scene; 0 or 1")] = None,
    action_type: Annotated[Optional[str], typer.Option(help="Action space type; 'discrete' or 'continuous'")] = None,
    use_vbd: Annotated[Optional[bool], typer.Option(help="Use VBD model for trajectory predictions")] = False,
    vbd_model_path: Annotated[Optional[str], typer.Option(help="Path to VBD model checkpoint")] = None,
    vbd_trajectory_weight: Annotated[Optional[float], typer.Option(help="Weight for VBD trajectory deviation penalty")] = 0.1,
    vbd_in_obs: Annotated[Optional[bool], typer.Option(help="Include VBD predictions in the observation")] = False,
    init_steps: Annotated[Optional[int], typer.Option(help="Environment warmup steps")] = 0,
    # Train options
    seed: Annotated[Optional[int], typer.Option(help="The seed for training")] = None,
    learning_rate: Annotated[Optional[float], typer.Option(help="The learning rate for training")] = None,
    anneal_lr: Annotated[Optional[int], typer.Option(help="Whether to anneal the learning rate over time; 0 or 1")] = None,
    resample_scenes: Annotated[Optional[int], typer.Option(help="Whether to resample scenes during training; 0 or 1")] = None,
    resample_interval: Annotated[Optional[int], typer.Option(help="The interval for resampling scenes")] = None,
    resample_dataset_size: Annotated[Optional[int], typer.Option(help="The size of the dataset to sample from")] = None,
    total_timesteps: Annotated[Optional[int], typer.Option(help="The total number of training steps")] = None,
    ent_coef: Annotated[Optional[float], typer.Option(help="Entropy coefficient")] = None,
    update_epochs: Annotated[Optional[int], typer.Option(help="The number of epochs for updating the policy")] = None,
    batch_size: Annotated[Optional[int], typer.Option(help="The batch size for training")] = None,
    minibatch_size: Annotated[Optional[int], typer.Option(help="The minibatch size for training")] = None,
    gamma: Annotated[Optional[float], typer.Option(help="The discount factor for rewards")] = None,
    vf_coef: Annotated[Optional[float], typer.Option(help="Weight for vf_loss")] = None,
    # GAIL-specific options
    use_gail: Annotated[Optional[bool], typer.Option(help="Use GAIL instead of regular PPO")] = None,
    discriminator_lr: Annotated[Optional[float], typer.Option(help="Learning rate for discriminator")] = None,
    discriminator_hidden_dim: Annotated[Optional[int], typer.Option(help="Hidden dimension for discriminator")] = None,
    discriminator_dropout: Annotated[Optional[float], typer.Option(help="Dropout rate for discriminator")] = None,
    discriminator_batch_size: Annotated[Optional[int], typer.Option(help="Batch size for discriminator training")] = None,
    discriminator_epochs: Annotated[Optional[int], typer.Option(help="Number of epochs to train discriminator per update")] = None,
    discriminator_update_freq: Annotated[Optional[int], typer.Option(help="Frequency of discriminator updates")] = None,
    policy_buffer_size: Annotated[Optional[int], typer.Option(help="Size of policy data buffer for discriminator training")] = None,
    min_policy_data: Annotated[Optional[int], typer.Option(help="Minimum policy data before training discriminator")] = None,
    # L2 Regularization options
    policy_weight_decay: Annotated[Optional[float], typer.Option(help="L2 regularization for policy network")] = None,
    value_weight_decay: Annotated[Optional[float], typer.Option(help="L2 regularization for value network")] = None,
    discriminator_weight_decay: Annotated[Optional[float], typer.Option(help="L2 regularization for discriminator")] = None,
    # Wandb logging options
    project: Annotated[Optional[str], typer.Option(help="WandB project name")] = None,
    entity: Annotated[Optional[str], typer.Option(help="WandB entity name")] = None,
    group: Annotated[Optional[str], typer.Option(help="WandB group name")] = None,
    render: Annotated[Optional[int], typer.Option(help="Whether to render the environment; 0 or 1")] = None,
    # Gradient penalty and stability improvements
    use_gradient_penalty: Annotated[Optional[bool], typer.Option(help="Whether to use gradient penalty")] = None,
    gradient_penalty_lambda: Annotated[Optional[float], typer.Option(help="Gradient penalty lambda")] = None,
    grad_clip_norm: Annotated[Optional[float], typer.Option(help="Gradient clipping norm")] = None,
    disc_updates_per_policy_update: Annotated[Optional[int], typer.Option(help="Discriminator updates per policy update")] = None,
    # Visualization options
    visualize: Annotated[Optional[bool], typer.Option(help="Whether to visualize the training")] = None,
    vis_interval: Annotated[Optional[int], typer.Option(help="Steps between visualizations")] = None,
    vis_rollout_length: Annotated[Optional[int], typer.Option(help="Length of rollouts for visualization")] = None,
    vis_num_rollouts: Annotated[Optional[int], typer.Option(help="Number of parallel rollouts for visualization")] = None,
):
    """Run PPO or GAIL training with the given configuration."""
    # fmt: on

    # Load default configs
    config = load_config(config_path)

    # Override configs with command-line arguments
    env_config = {
        "num_worlds": num_worlds,
        "k_unique_scenes": k_unique_scenes,
        "collision_weight": collision_weight,
        "off_road_weight": off_road_weight,
        "goal_achieved_weight": goal_achieved_weight,
        "dist_to_goal_threshold": dist_to_goal_threshold,
        "sampling_seed": sampling_seed,
        "obs_radius": obs_radius,
        "collision_behavior": collision_behavior,
        "remove_non_vehicles": None
        if remove_non_vehicles is None
        else bool(remove_non_vehicles),
        "action_type": action_type,
        "use_vbd": use_vbd,
        "vbd_model_path": vbd_model_path,
        "vbd_trajectory_weight": vbd_trajectory_weight,
        "vbd_in_obs": vbd_in_obs,
        "init_steps": init_steps,
    }
    config.environment.update(
        {k: v for k, v in env_config.items() if v is not None}
    )

    train_config = {
        "seed": seed,
        "learning_rate": learning_rate,
        "anneal_lr": None if anneal_lr is None else bool(anneal_lr),
        "resample_scenes": None
        if resample_scenes is None
        else bool(resample_scenes),
        "resample_interval": resample_interval,
        "resample_dataset_size": resample_dataset_size,
        "total_timesteps": total_timesteps,
        "ent_coef": ent_coef,
        "update_epochs": update_epochs,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "render": None if render is None else bool(render),
        "gamma": gamma,
        "vf_coef": vf_coef,
        # GAIL parameters with defaults
        "use_gail": use_gail if use_gail is not None else False,
        "discriminator_lr": discriminator_lr,
        "discriminator_hidden_dim": discriminator_hidden_dim,
        "discriminator_dropout": discriminator_dropout,
        "discriminator_batch_size": discriminator_batch_size,
        "discriminator_epochs": discriminator_epochs,
        "discriminator_update_freq": discriminator_update_freq,
        "policy_buffer_size": policy_buffer_size,
        "min_policy_data": min_policy_data,
        # L2 regularization parameters
        "policy_weight_decay": policy_weight_decay,
        "value_weight_decay": value_weight_decay,
        "discriminator_weight_decay": discriminator_weight_decay,
        # NEW: Gradient penalty and stability improvements
        "use_gradient_penalty": use_gradient_penalty,  # Enable gradient penalty by default
        "gradient_penalty_lambda": gradient_penalty_lambda,  # Standard WGAN-GP coefficient
        "grad_clip_norm": grad_clip_norm,  # Gradient clipping for additional stability
        "disc_updates_per_policy_update": disc_updates_per_policy_update,  # Best practice: 1:1 or 2:1 ratio
        # Visualization parameters
        "visualize": visualize,
        "vis_interval": vis_interval,
        "vis_rollout_length": vis_rollout_length,
        "vis_num_rollouts": vis_num_rollouts,
    }
    config.train.update(
        {k: v for k, v in train_config.items() if v is not None}
    )
    print(config.train.render)

    wandb_config = {
        "project": project,
        "entity": entity,
        "group": group,
    }
    config.wandb.update(
        {k: v for k, v in wandb_config.items() if v is not None}
    )

    datetime_ = datetime.now().strftime("%m_%d_%H_%M_%S_%f")[:-3]

    if config["continue_training"]:
        cont_train = "C"
    else:
        cont_train = ""

    # Update experiment ID to include GAIL identifier
    method_prefix = "GAIL" if config.train.use_gail else "PPO"
    
    if config["train"]["resample_scenes"]:
        if config["train"]["resample_scenes"]:
            dataset_size = config["train"]["resample_dataset_size"]
        config["train"][
            "exp_id"
        ] = f'{method_prefix}_{config["train"]["exp_id"]}__{cont_train}__R_{dataset_size}__{datetime_}'
    else:
        dataset_size = str(config["environment"]["k_unique_scenes"])
        config["train"][
            "exp_id"
        ] = f'{method_prefix}_{config["train"]["exp_id"]}__{cont_train}__S_{dataset_size}__{datetime_}'

    config["environment"]["dataset_size"] = dataset_size
    config["train"]["device"] = config["train"].get(
        "device", "cpu"
    )  # Default to 'cpu' if not set
    if torch.cuda.is_available():
        print("Using GPU")
        config["train"]["device"] = "cuda"  # Set to 'cuda' if available

    # Make dataloader
    train_loader = SceneDataLoader(
        root=config.data_dir,
        batch_size=config.environment.num_worlds,
        dataset_size=config.train.resample_dataset_size
        if config.train.resample_scenes
        else config.environment.k_unique_scenes,
        sample_with_replacement=config.train.sample_with_replacement,
        shuffle=config.train.shuffle_dataset,
        seed=seed if seed is not None else 42,
    )

    # Print action type for debugging
    action_type = getattr(config.environment, 'action_type', 'discrete')
    print(f"Action Type: {action_type.upper()}")
    
    # Make environment AFTER wandb config is available
    vecenv = PufferGPUDrive(
        data_loader=train_loader,
        **config.environment,
        **config.train,
    )

    # print(f"vecenv.single_observation_space: {vecenv.single_observation_space}")

    train_gail(config, vecenv)


if __name__ == "__main__":

    app()
