"""
Single-Agent GAIL Implementation
Creates individual single-agent environments from multi-agent scenarios.
Each environment has exactly one controlled agent with others following original trajectories.
"""

import os
from typing import Optional, List, Tuple
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

from gpudrive.integrations.puffer import ppo
from gpudrive.env.env_puffer import PufferGPUDrive
from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.dataset import SceneDataLoader
from baselines.imitation_data_generation import generate_state_action_pairs

import pufferlib
import pufferlib.vector
import pufferlib.cleanrl
from rich.console import Console

import typer
from typer import Typer

# Import from main gail module
from irl.gail import (
    Discriminator, 
    ExpertDataset, 
    GAILTrainer, 
    get_model_parameters,
    load_config,
    init_wandb
)

app = Typer()


class SingleAgentEnvironmentWrapper:
    """
    Wrapper that converts multi-agent environments into single-agent environments.
    Each 'world' becomes an individual single-agent environment.
    """
    
    def __init__(self, base_env, max_single_agent_envs=None):
        self.base_env = base_env
        
        # Get the original controlled agent mask
        self.original_controlled_mask = base_env.cont_agent_mask.clone()
        
        # Find all controlled agents across all worlds
        self.controlled_agent_positions = []
        for world_idx in range(self.original_controlled_mask.shape[0]):
            for agent_idx in range(self.original_controlled_mask.shape[1]):
                if self.original_controlled_mask[world_idx, agent_idx]:
                    self.controlled_agent_positions.append((world_idx, agent_idx))
        
        # Limit number of single-agent environments if specified
        if max_single_agent_envs and max_single_agent_envs < len(self.controlled_agent_positions):
            self.controlled_agent_positions = self.controlled_agent_positions[:max_single_agent_envs]
        
        self.num_single_agent_envs = len(self.controlled_agent_positions)
        print(f"Created {self.num_single_agent_envs} single-agent environments")
        
        # Create new controlled mask: each row is a single-agent environment
        self.single_agent_controlled_mask = torch.zeros(
            (self.num_single_agent_envs, self.original_controlled_mask.shape[1]),
            dtype=torch.bool
        )
        
        # Set exactly one agent as controlled in each environment
        for env_idx, (world_idx, agent_idx) in enumerate(self.controlled_agent_positions):
            self.single_agent_controlled_mask[env_idx, agent_idx] = True
        
        # Update environment properties
        self.cont_agent_mask = self.single_agent_controlled_mask
        self.num_worlds = self.num_single_agent_envs
        self.total_agents = self.num_single_agent_envs  # One agent per environment
        
        # Store mapping for observations and actions
        self.world_agent_mapping = self.controlled_agent_positions
        
    def reset(self):
        """Reset and return observations for single-agent environments."""
        # Reset the base environment
        base_obs = self.base_env.reset()
        
        # Extract observations for each single-agent environment
        single_agent_obs = []
        for env_idx, (world_idx, agent_idx) in enumerate(self.world_agent_mapping):
            obs = base_obs[world_idx, agent_idx:agent_idx+1, :]  # Keep batch dim
            single_agent_obs.append(obs)
        
        return torch.cat(single_agent_obs, dim=0)
    
    def step(self, actions):
        """Step the environment with single-agent actions."""
        # Convert single-agent actions back to multi-agent format
        full_actions = torch.zeros(
            (self.base_env.num_worlds, self.base_env.cont_agent_mask.shape[1]),
            dtype=actions.dtype,
            device=actions.device
        )
        
        for env_idx, (world_idx, agent_idx) in enumerate(self.world_agent_mapping):
            if env_idx < len(actions):
                full_actions[world_idx, agent_idx] = actions[env_idx]
        
        # Step the base environment
        return self.base_env.step(full_actions)
    
    def get_obs(self, mask=None):
        """Get observations for single-agent environments."""
        if mask is None:
            mask = self.cont_agent_mask
            
        # Get base observations
        base_obs = self.base_env.get_obs()
        
        # Extract observations for controlled agents
        single_agent_obs = []
        for env_idx, (world_idx, agent_idx) in enumerate(self.world_agent_mapping):
            obs = base_obs[world_idx, agent_idx:agent_idx+1, :]
            single_agent_obs.append(obs)
        
        return torch.cat(single_agent_obs, dim=0)
    
    def get_expert_actions(self):
        """Get expert actions formatted for single-agent environments."""
        # Get base expert actions
        base_expert_actions, speeds, positions, yaws = self.base_env.get_expert_actions()
        
        # Extract expert actions for each controlled agent
        single_agent_expert_actions = []
        for env_idx, (world_idx, agent_idx) in enumerate(self.world_agent_mapping):
            expert_actions = base_expert_actions[world_idx, agent_idx:agent_idx+1, :, :]
            single_agent_expert_actions.append(expert_actions)
        
        return (
            torch.cat(single_agent_expert_actions, dim=0),
            speeds, positions, yaws
        )
    
    def __getattr__(self, name):
        """Delegate other attributes to base environment."""
        return getattr(self.base_env, name)


def load_single_agent_expert_data(config, base_env, max_single_agent_envs=None):
    """Load expert data for single-agent GAIL training."""
    print("Generating single-agent expert demonstrations...")
    
    # Create single-agent wrapper
    single_agent_env = SingleAgentEnvironmentWrapper(base_env, max_single_agent_envs)
    
    # Generate expert data using the wrapper
    expert_obs, expert_actions, _, _, goal_rate, collision_rate = generate_state_action_pairs(
        env=single_agent_env,
        device=config.device,
        action_space_type="discrete" if hasattr(base_env, 'single_action_space') and hasattr(base_env.single_action_space, 'n') else "continuous",
        use_action_indices=True,
        make_video=False,
    )
    
    print(f"Single-agent expert data: {len(expert_obs)} transitions from {single_agent_env.num_single_agent_envs} agents")
    print(f"Goal rate: {goal_rate:.3f}, Collision rate: {collision_rate:.3f}")
    
    return expert_obs, expert_actions, single_agent_env


class SingleAgentVecEnv:
    """
    Vectorized environment wrapper for single-agent GAIL.
    Manages multiple single-agent environments efficiently.
    """
    
    def __init__(self, data_loader, single_agent_env, **config):
        self.data_loader = data_loader
        self.single_agent_env = single_agent_env
        self.driver_env = single_agent_env  # For compatibility
        
        # Environment properties
        self.num_worlds = single_agent_env.num_single_agent_envs
        self.total_agents = single_agent_env.num_single_agent_envs
        self.cont_agent_mask = single_agent_env.cont_agent_mask
        
        # Action/observation spaces
        self.single_action_space = single_agent_env.single_action_space
        self.single_observation_space = single_agent_env.single_observation_space
        
        # For PufferLib compatibility
        self.use_vbd = getattr(single_agent_env, 'use_vbd', False)
        
        print(f"SingleAgentVecEnv created with {self.num_worlds} single-agent environments")
    
    def async_reset(self, seed):
        """Reset all single-agent environments."""
        return self.single_agent_env.reset()
    
    def recv(self):
        """Receive step results from environments."""
        # This would be implemented based on your specific environment interface
        # For now, return placeholder values
        obs = self.single_agent_env.get_obs()
        reward = torch.zeros(self.num_worlds)
        terminal = torch.zeros(self.num_worlds, dtype=torch.bool)
        truncated = torch.zeros(self.num_worlds, dtype=torch.bool)
        info = {}
        env_id = torch.arange(self.num_worlds)
        mask = torch.ones(self.num_worlds, dtype=torch.bool)
        
        return obs, reward, terminal, truncated, info, env_id, mask
    
    def send(self, actions):
        """Send actions to environments."""
        return self.single_agent_env.step(actions)
    
    def resample_scenario_batch(self):
        """Resample scenarios if needed."""
        if hasattr(self.single_agent_env.base_env, 'swap_data_batch'):
            self.single_agent_env.base_env.swap_data_batch()
    
    def clear_render_storage(self):
        """Clear render storage."""
        pass
    
    def log_data_coverage(self):
        """Log data coverage."""
        pass
    
    def __getattr__(self, name):
        """Delegate to single agent environment."""
        return getattr(self.single_agent_env, name)


def train_single_agent_gail(args, base_vecenv, max_single_agent_envs=None):
    """Main single-agent GAIL training loop."""
    
    # Load expert data and create single-agent environment
    expert_obs, expert_actions, single_agent_env = load_single_agent_expert_data(
        args.train, base_vecenv.driver_env, max_single_agent_envs
    )
    
    # Create single-agent vectorized environment
    vecenv = SingleAgentVecEnv(base_vecenv.data_loader, single_agent_env, **args.environment)
    
    # Create policy (generator)
    policy = make_agent(env=vecenv.driver_env, config=args).to(args.train.device)
    args.train.network.num_parameters = get_model_parameters(policy)
    args.train.env = args.environment.name

    # Initialize wandb
    args.wandb = init_wandb(args, args.train.exp_id, id=args.train.exp_id)
    args.train.__dict__.update(dict(args.wandb.config.train))

    # Create expert dataset
    expert_dataset = ExpertDataset(expert_obs, expert_actions, device=args.train.device)
    
    # Create discriminator
    obs_dim = expert_obs.shape[-1]
    action_dim = 1 if expert_actions.dim() == 1 else expert_actions.shape[-1]
    discriminator = Discriminator(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.train.discriminator_hidden_dim,
        dropout=args.train.discriminator_dropout
    ).to(args.train.device)
    
    # Create PPO trainer
    ppo_data = ppo.create(args.train, vecenv, policy, wandb=args.wandb)
    
    # Apply L2 regularization if specified
    policy_weight_decay = getattr(args.train, 'policy_weight_decay', 0.0)
    value_weight_decay = getattr(args.train, 'value_weight_decay', 0.0)
    
    if policy_weight_decay > 0 or value_weight_decay > 0:
        if hasattr(policy, 'actor') and hasattr(policy, 'critic'):
            policy_params = list(policy.actor.parameters())
            value_params = list(policy.critic.parameters())
        else:
            policy_params = list(policy.parameters())
            value_params = []
        
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
            ppo_data.optimizer = torch.optim.Adam(
                policy.parameters(), 
                lr=float(args.train.learning_rate), 
                eps=1e-5, 
                weight_decay=policy_weight_decay
            )
            print(f"Applied L2 regularization to all parameters: {policy_weight_decay}")
    
    # Create GAIL trainer
    gail_trainer = GAILTrainer(args.train, discriminator, expert_dataset, ppo_data)
    
    print(f"Policy parameters: {get_model_parameters(policy):,}")
    print(f"Discriminator parameters: {get_model_parameters(discriminator):,}")
    print(f"Training on {vecenv.num_worlds} single-agent environments")
    
    step_count = 0
    while ppo_data.global_step < args.train.total_timesteps:
        try:
            # Standard PPO rollout but collect data for discriminator
            ppo.evaluate(ppo_data)
            
            # Extract rollout data for discriminator training
            obs_data = ppo_data.experience.obs[:ppo_data.experience.ptr].flatten(0, 1)
            action_data = ppo_data.experience.action[:ppo_data.experience.ptr].flatten(0, 1)
            
            # Add policy data to GAIL trainer
            gail_trainer.add_policy_data(obs_data, action_data)
            
            # Replace environment rewards with discriminator rewards
            with torch.no_grad():
                disc_rewards = discriminator.predict_reward(
                    obs_data.to(args.train.device), 
                    action_data.to(args.train.device)
                )
                # Reshape to match experience buffer format
                disc_rewards = disc_rewards.view(ppo_data.experience.reward[:ppo_data.experience.ptr].shape)
                ppo_data.experience.reward[:ppo_data.experience.ptr] = disc_rewards.cpu()
            
            # Train discriminator every few steps
            if step_count % args.train.discriminator_update_freq == 0:
                gail_trainer.train_discriminator(args.train.discriminator_epochs)
            
            # Train policy with discriminator rewards
            ppo.train(ppo_data)
            
            step_count += 1
            
        except KeyboardInterrupt:
            ppo.close(ppo_data)
            os._exit(0)
        except Exception as e:
            print(f"An error occurred: {e}")
            Console().print_exception()
            os._exit(1)

    ppo.evaluate(ppo_data)
    ppo.close(ppo_data)


def make_agent(env, config):
    """Create a policy based on the environment."""
    if config.continue_training:
        print("Loading checkpoint...")
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
        policy.load_state_dict(saved_cpt["parameters"])
        return policy
    else:
        return NeuralNet(
            input_dim=config.train.network.input_dim,
            action_dim=env.single_action_space.n,
            hidden_dim=config.train.network.hidden_dim,
            dropout=config.train.network.dropout,
            config=config.environment,
        )


@app.command()
def run(
    config_path: Annotated[
        str, typer.Argument(help="The path to the configuration file")
    ],
    *,
    # Environment options
    num_worlds: Annotated[Optional[int], typer.Option(help="Number of parallel worlds (for base env)")] = None,
    k_unique_scenes: Annotated[Optional[int], typer.Option(help="The number of unique scenes to sample")] = None,
    max_single_agent_envs: Annotated[Optional[int], typer.Option(help="Maximum number of single-agent environments to create")] = None,
    collision_weight: Annotated[Optional[float], typer.Option(help="The weight for collision penalty")] = None,
    off_road_weight: Annotated[Optional[float], typer.Option(help="The weight for off-road penalty")] = None,
    goal_achieved_weight: Annotated[Optional[float], typer.Option(help="The weight for goal-achieved reward")] = None,
    dist_to_goal_threshold: Annotated[Optional[float], typer.Option(help="The distance threshold for goal-achieved")] = None,
    sampling_seed: Annotated[Optional[int], typer.Option(help="The seed for sampling scenes")] = None,
    obs_radius: Annotated[Optional[float], typer.Option(help="The radius for the observation")] = None,
    collision_behavior: Annotated[Optional[str], typer.Option(help="The collision behavior; 'ignore' or 'remove'")] = None,
    remove_non_vehicles: Annotated[Optional[int], typer.Option(help="Remove non-vehicles from the scene; 0 or 1")] = None,
    # Train options
    seed: Annotated[Optional[int], typer.Option(help="The seed for training")] = None,
    learning_rate: Annotated[Optional[float], typer.Option(help="The learning rate for training")] = None,
    total_timesteps: Annotated[Optional[int], typer.Option(help="The total number of training steps")] = None,
    batch_size: Annotated[Optional[int], typer.Option(help="The batch size for training")] = None,
    minibatch_size: Annotated[Optional[int], typer.Option(help="The minibatch size for training")] = None,
    # GAIL-specific options
    discriminator_lr: Annotated[Optional[float], typer.Option(help="Learning rate for discriminator")] = None,
    discriminator_hidden_dim: Annotated[Optional[int], typer.Option(help="Hidden dimension for discriminator")] = None,
    discriminator_epochs: Annotated[Optional[int], typer.Option(help="Number of epochs to train discriminator per update")] = None,
    policy_weight_decay: Annotated[Optional[float], typer.Option(help="L2 regularization for policy network")] = None,
    discriminator_weight_decay: Annotated[Optional[float], typer.Option(help="L2 regularization for discriminator")] = None,
    # Wandb logging options
    project: Annotated[Optional[str], typer.Option(help="WandB project name")] = None,
    entity: Annotated[Optional[str], typer.Option(help="WandB entity name")] = None,
    group: Annotated[Optional[str], typer.Option(help="WandB group name")] = None,
):
    """Run single-agent GAIL training."""
    
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
        "remove_non_vehicles": None if remove_non_vehicles is None else bool(remove_non_vehicles),
    }
    config.environment.update({k: v for k, v in env_config.items() if v is not None})

    train_config = {
        "seed": seed,
        "learning_rate": learning_rate,
        "total_timesteps": total_timesteps,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "discriminator_lr": discriminator_lr if discriminator_lr is not None else 3e-4,
        "discriminator_hidden_dim": discriminator_hidden_dim if discriminator_hidden_dim is not None else 256,
        "discriminator_epochs": discriminator_epochs if discriminator_epochs is not None else 5,
        "policy_weight_decay": policy_weight_decay if policy_weight_decay is not None else 1e-4,
        "discriminator_weight_decay": discriminator_weight_decay if discriminator_weight_decay is not None else 1e-4,
        "max_single_agent_envs": max_single_agent_envs,
    }
    config.train.update({k: v for k, v in train_config.items() if v is not None})

    wandb_config = {
        "project": project,
        "entity": entity,
        "group": group,
    }
    config.wandb.update({k: v for k, v in wandb_config.items() if v is not None})

    # Update experiment ID
    datetime_ = datetime.now().strftime("%m_%d_%H_%M_%S_%f")[:-3]
    max_envs_str = f"_{max_single_agent_envs}" if max_single_agent_envs else ""
    config.train.exp_id = f"SingleAgent_GAIL{max_envs_str}__{datetime_}"

    # Set device
    config.train.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {config.train.device}")

    # Make base dataloader
    train_loader = SceneDataLoader(
        root=config.data_dir,
        batch_size=config.environment.num_worlds,
        dataset_size=config.environment.k_unique_scenes,
        sample_with_replacement=getattr(config.train, 'sample_with_replacement', True),
        shuffle=getattr(config.train, 'shuffle_dataset', False),
        seed=seed if seed is not None else 42,
    )

    # Make base environment
    base_vecenv = PufferGPUDrive(
        data_loader=train_loader,
        **config.environment,
        **config.train,
    )

    print("Starting Single-Agent GAIL training...")
    train_single_agent_gail(config, base_vecenv, max_single_agent_envs)


if __name__ == "__main__":
    app() 