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

app = Typer()

def get_model_parameters(policy):
    """Helper function to count the number of trainable parameters."""
    params = filter(lambda p: p.requires_grad, policy.parameters())
    return sum([np.prod(p.size()) for p in params])


class Discriminator(nn.Module):
    """Discriminator network for GAIL that distinguishes expert from policy trajectories."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256, dropout=0.0):
        super().__init__()
        
        # Input is state-action pair
        input_dim = obs_dim + action_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, states, actions):
        """
        Forward pass of discriminator.
        
        Args:
            states: Tensor of shape (batch_size, obs_dim)
            actions: Tensor of shape (batch_size, action_dim) or (batch_size,) for discrete
            
        Returns:
            logits: Tensor of shape (batch_size, 1) - logit for real (expert) vs fake (policy)
        """
        # Handle discrete actions
        if actions.dim() == 1:
            # Convert discrete actions to one-hot if needed
            if actions.dtype in (torch.long, torch.int):
                actions = F.one_hot(actions, num_classes=self.network[0].in_features - states.shape[-1]).float()
            else:
                actions = actions.unsqueeze(-1)
        
        # Concatenate state and action
        state_action = torch.cat([states, actions], dim=-1)
        return self.network(state_action)
    
    def predict_reward(self, states, actions):
        """
        Predict reward based on discriminator output.
        In GAIL, reward is -log(1 - D(s,a)) where D(s,a) is discriminator output.
        """
        with torch.no_grad():
            logits = self.forward(states, actions)
            # Convert logits to probability of being expert
            prob_expert = torch.sigmoid(logits)
            # GAIL reward: -log(1 - D(s,a))
            reward = -torch.log(1 - prob_expert + 1e-8)
            return reward.squeeze(-1)


class ExpertDataset:
    """Dataset for storing and sampling expert trajectories."""
    
    def __init__(self, expert_obs, expert_actions, device='cpu'):
        self.expert_obs = expert_obs.to(device)
        self.expert_actions = expert_actions.to(device)
        self.device = device
        self.size = len(expert_obs)
        
    def sample(self, batch_size):
        """Sample a batch of expert state-action pairs."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.expert_obs[indices], self.expert_actions[indices]


class GAILTrainer:
    """GAIL training coordinator that manages alternating discriminator and policy training."""
    
    def __init__(self, config, discriminator, expert_dataset, policy_trainer_data):
        self.config = config
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
        self.policy_action_buffer = deque(maxlen=config.policy_buffer_size)
        
        # Logging
        self.disc_losses = []
        self.disc_expert_acc = []
        self.disc_policy_acc = []
        
    def add_policy_data(self, obs, actions):
        """Add policy-generated trajectories to buffer."""
        self.policy_obs_buffer.extend(obs.cpu())
        self.policy_action_buffer.extend(actions.cpu())
    
    def train_discriminator(self, num_epochs=5):
        """Train discriminator to distinguish expert from policy data."""
        if len(self.policy_obs_buffer) < self.config.min_policy_data:
            return
            
        disc_losses = []
        expert_accs = []
        policy_accs = []
        
        for epoch in range(num_epochs):
            # Sample expert data
            expert_obs, expert_actions = self.expert_dataset.sample(self.config.discriminator_batch_size // 2)
            
            # Sample policy data
            if len(self.policy_obs_buffer) >= self.config.discriminator_batch_size // 2:
                policy_indices = random.sample(range(len(self.policy_obs_buffer)), self.config.discriminator_batch_size // 2)
                policy_obs = torch.stack([self.policy_obs_buffer[i] for i in policy_indices]).to(self.config.device)
                policy_actions = torch.stack([self.policy_action_buffer[i] for i in policy_indices]).to(self.config.device)
            else:
                # Use all available policy data and pad with expert data
                policy_obs = torch.stack(list(self.policy_obs_buffer)).to(self.config.device)
                policy_actions = torch.stack(list(self.policy_action_buffer)).to(self.config.device)
                
                remaining = self.config.discriminator_batch_size // 2 - len(policy_obs)
                if remaining > 0:
                    extra_expert_obs, extra_expert_actions = self.expert_dataset.sample(remaining)
                    policy_obs = torch.cat([policy_obs, extra_expert_obs])
                    policy_actions = torch.cat([policy_actions, extra_expert_actions])
            
            # Create labels: 1 for expert, 0 for policy
            expert_labels = torch.ones(len(expert_obs), 1, device=self.config.device)
            policy_labels = torch.zeros(len(policy_obs), 1, device=self.config.device)
            
            # Combine data
            all_obs = torch.cat([expert_obs, policy_obs])
            all_actions = torch.cat([expert_actions, policy_actions])
            all_labels = torch.cat([expert_labels, policy_labels])
            
            # Forward pass
            self.disc_optimizer.zero_grad()
            logits = self.discriminator(all_obs, all_actions)
            
            # Binary cross entropy loss
            loss = F.binary_cross_entropy_with_logits(logits, all_labels)
            
            # Backward pass
            loss.backward()
            self.disc_optimizer.step()
            
            # Calculate accuracies
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).float()
                
                expert_acc = (predictions[:len(expert_obs)] == expert_labels).float().mean()
                policy_acc = (predictions[len(expert_obs):] == policy_labels).float().mean()
                
                disc_losses.append(loss.item())
                expert_accs.append(expert_acc.item())
                policy_accs.append(policy_acc.item())
        
        # Store averages
        self.disc_losses.append(np.mean(disc_losses))
        self.disc_expert_acc.append(np.mean(expert_accs))
        self.disc_policy_acc.append(np.mean(policy_accs))
        
        # Log to wandb
        if hasattr(self.policy_trainer_data, 'wandb') and self.policy_trainer_data.wandb:
            self.policy_trainer_data.wandb.log({
                'discriminator/loss': self.disc_losses[-1],
                'discriminator/expert_accuracy': self.disc_expert_acc[-1],
                'discriminator/policy_accuracy': self.disc_policy_acc[-1],
                'global_step': self.policy_trainer_data.global_step
            })


def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return pufferlib.namespace(**config)


def load_expert_data(config, env):
    """Load expert demonstrations for GAIL training."""
    print("Generating expert demonstrations...")
    
    # Generate expert data using the existing function
    expert_obs, expert_actions, _, _, goal_rate, collision_rate = generate_state_action_pairs(
        env=env,
        device=config.device,
        action_space_type="discrete" if hasattr(env, 'single_action_space') and hasattr(env.single_action_space, 'n') else "continuous",
        use_action_indices=True,
        make_video=False,
    )
    
    print(f"Expert data: {len(expert_obs)} transitions, Goal rate: {goal_rate:.3f}, Collision rate: {collision_rate:.3f}")
    
    return expert_obs, expert_actions


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
        return NeuralNet(
            input_dim=config.train.network.input_dim,
            action_dim=env.single_action_space.n,
            hidden_dim=config.train.network.hidden_dim,
            dropout=config.train.network.dropout,
            config=config.environment,
        )


def train_gail(args, vecenv):
    """Main GAIL training loop."""
    # Create policy (generator)
    policy = make_agent(env=vecenv.driver_env, config=args).to(args.train.device)
    args.train.network.num_parameters = get_model_parameters(policy)
    args.train.env = args.environment.name

    # Initialize wandb
    args.wandb = init_wandb(args, args.train.exp_id, id=args.train.exp_id)
    args.train.__dict__.update(dict(args.wandb.config.train))

    # Load expert data
    expert_obs, expert_actions = load_expert_data(args.train, vecenv.driver_env)
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
    gail_trainer = GAILTrainer(args.train, discriminator, expert_dataset, ppo_data)
    
    print(f"Policy parameters: {get_model_parameters(policy):,}")
    print(f"Discriminator parameters: {get_model_parameters(discriminator):,}")
    
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


def sweep(args, project="PPO", sweep_name="my_sweep"):
    """Initialize a WandB sweep with hyperparameters."""
    sweep_id = wandb.sweep(
        sweep=dict(
            method="random",
            name=sweep_name,
            metric={"goal": "maximize", "name": "environment/episode_return"},
            parameters={
                "learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 1e-4,
                    "max": 1e-1,
                },
                "batch_size": {"values": [512, 1024, 2048]},
                "minibatch_size": {"values": [128, 256, 512]},
            },
        ),
        project=project,
    )
    wandb.agent(sweep_id, lambda: train(args), count=100)


@app.command()
def run(
    config_path: Annotated[
        str, typer.Argument(help="The path to the configuration file")
    ],
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
    use_gail: Annotated[Optional[bool], typer.Option(help="Use GAIL instead of regular PPO")] = False,
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
        "discriminator_lr": discriminator_lr if discriminator_lr is not None else 3e-4,
        "discriminator_hidden_dim": discriminator_hidden_dim if discriminator_hidden_dim is not None else 256,
        "discriminator_dropout": discriminator_dropout if discriminator_dropout is not None else 0.1,
        "discriminator_batch_size": discriminator_batch_size if discriminator_batch_size is not None else 256,
        "discriminator_epochs": discriminator_epochs if discriminator_epochs is not None else 5,
        "discriminator_update_freq": discriminator_update_freq if discriminator_update_freq is not None else 1,
        "policy_buffer_size": policy_buffer_size if policy_buffer_size is not None else 10000,
        "min_policy_data": min_policy_data if min_policy_data is not None else 1000,
        # L2 regularization parameters
        "policy_weight_decay": policy_weight_decay if policy_weight_decay is not None else 1e-4,
        "value_weight_decay": value_weight_decay if value_weight_decay is not None else 1e-4,
        "discriminator_weight_decay": discriminator_weight_decay if discriminator_weight_decay is not None else 1e-4,
    }
    config.train.update(
        {k: v for k, v in train_config.items() if v is not None}
    )

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

    # Make environment
    vecenv = PufferGPUDrive(
        data_loader=train_loader,
        **config.environment,
        **config.train,
    )

    # Choose training method
    if config.train.use_gail:
        print("Starting GAIL training...")
        train_gail(config, vecenv)
    else:
        print("Starting regular PPO training...")
        train_ppo(config, vecenv)


def train_ppo(args, vecenv):
    """Main training loop for regular PPO agent."""
    policy = make_agent(env=vecenv.driver_env, config=args).to(
        args.train.device
    )

    args.train.network.num_parameters = get_model_parameters(policy)
    args.train.env = args.environment.name

    args.wandb = init_wandb(args, args.train.exp_id, id=args.train.exp_id)
    args.train.__dict__.update(dict(args.wandb.config.train))

    data = ppo.create(args.train, vecenv, policy, wandb=args.wandb)
    
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
            data.optimizer = torch.optim.Adam(param_groups, eps=1e-5)
            print(f"Applied L2 regularization - Policy: {policy_weight_decay}, Value: {value_weight_decay}")
        else:
            data.optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=float(args.train.learning_rate),
                eps=1e-5,
                weight_decay=policy_weight_decay
            )
            print(f"Applied L2 regularization to all parameters: {policy_weight_decay}")
    while data.global_step < args.train.total_timesteps:
        try:
            ppo.evaluate(data)  # Rollout
            ppo.train(data)  # Update policy
        except KeyboardInterrupt:
            ppo.close(data)
            os._exit(0)
        except Exception as e:
            print(f"An error occurred: {e}")  # Log the error
            Console().print_exception()
            os._exit(1)  # Exit with a non-zero status to indicate an error

    ppo.evaluate(data)
    ppo.close(data)


if __name__ == "__main__":

    app()
