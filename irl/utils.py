import torch
import numpy as np
import yaml
from box import Box
import pufferlib
from gpudrive.networks.late_fusion import NeuralNet
import torch.nn as nn
import torch.nn.functional as F
import wandb
from storage import save_trajectory
import os

class ExpertDataset:
    def __init__(self, expert_obs, config):
        # Reshape to (num_samples*num_steps, obs_dim)
        self.expert_obs = expert_obs.reshape(-1, expert_obs.shape[-1])
        self.batch_size = min(len(expert_obs), config.train.batch_size)
        print(f"Expert observations shape: {expert_obs.shape}, batch size: {self.batch_size}")
    
    def next_iter(self):
        """Get a randomly shuffled batch from the expert dataset."""
        # Randomly sample batch_size indices from the entire expert dataset
        total_samples = self.expert_obs.shape[0]
        random_indices = torch.randperm(total_samples)[:self.batch_size]
        return self.expert_obs[random_indices]

class Discriminator(nn.Module):
    """Discriminator network for GAIL that distinguishes expert from policy trajectories based on states only."""
    
    def __init__(self, obs_dim, hidden_dim=64, dropout=0.0, config=None):
        super().__init__()
        
        # Store dimensions for later use
        self.obs_dim = obs_dim
        
        # Input is state only
        input_dim = obs_dim
        
        # Create network layers
        layers = []
        
        # Add input normalization layer if specified
        if config and config.gail.discriminator_use_input_norm:
            layers.append(nn.LayerNorm(input_dim))
        
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, states):
        """
        Forward pass of discriminator.
        """
        assert states.dim() == 2, "States must be 2D"
        assert states.shape[1] == self.obs_dim, "States must have correct dimension"
        return self.network(states)

    def predict_reward(self, states, reward_type="classic"):
        """
        Predict reward based on the discriminator's output.
        Args:
            states: Tensor of shape (batch_size, obs_dim)
        """
        assert reward_type in ["classic", "AIRL", "negativelnD"], "Supported reward types: classic, AIRL, negativelnD"
        logits = self.forward(states)
        if reward_type == "classic":
            reward = -F.logsigmoid(-logits)
        if reward_type == "AIRL":
            classification_reward = torch.sigmoid(logits)
            reward = torch.log(classification_reward) - torch.log(1 - classification_reward)
        if reward_type == "negativelnD":
            reward = -torch.log(1 - torch.sigmoid(logits))
        return reward



def train_discriminator(policy_data, expert_data, discriminator, optimizer, num_minibatches=8, test_accuracy=False):
    # Get the device from discriminator's parameters
    device = next(discriminator.parameters()).device
    
    policy_data = policy_data.to(device)
    expert_data = expert_data.to(device)

    expert_labels = torch.ones(expert_data.shape[0], 1, device=device)
    policy_labels = torch.zeros(policy_data.shape[0], 1, device=device)

    print(f"expert_data shape: {expert_data.shape}, policy_data shape: {policy_data.shape}")

    expert_minibatch_size = expert_data.shape[0] // num_minibatches
    policy_minibatch_size = policy_data.shape[0] // num_minibatches
    
    # Shuffle indices for both datasets
    expert_indices = torch.randperm(expert_data.shape[0], device=device)
    policy_indices = torch.randperm(policy_data.shape[0], device=device)
    
    for batch_idx in range(num_minibatches + 1):
        optimizer.zero_grad()

        # Process expert data batch
        expert_start = (batch_idx * expert_minibatch_size) % expert_data.shape[0]
        expert_end = min(expert_start + expert_minibatch_size, expert_data.shape[0])
        if expert_end <= expert_start:
            expert_end = expert_data.shape[0]
        
        expert_batch_indices = expert_indices[expert_start:expert_end]
        expert_batch = expert_data[expert_batch_indices]
        expert_batch_labels = expert_labels[expert_batch_indices]
        
        expert_logits = discriminator(expert_batch)
        expert_loss = F.binary_cross_entropy_with_logits(expert_logits, expert_batch_labels)
        
        # Process policy data batch
        policy_start = (batch_idx * policy_minibatch_size) % policy_data.shape[0]
        policy_end = min(policy_start + policy_minibatch_size, policy_data.shape[0])
        if policy_end <= policy_start:
            policy_end = policy_data.shape[0]
            
        policy_batch_indices = policy_indices[policy_start:policy_end]
        policy_batch = policy_data[policy_batch_indices]
        policy_batch_labels = policy_labels[policy_batch_indices]
        
        policy_logits = discriminator(policy_batch)
        policy_loss = F.binary_cross_entropy_with_logits(policy_logits, policy_batch_labels)
        
        # Combine losses with equal weighting
        combined_loss = 0.5 * expert_loss + 0.5 * policy_loss
        combined_loss.backward()
        optimizer.step()
    
    # test the accuracy of the discriminator
    if(test_accuracy):
        with torch.no_grad():
            # calculate the accuracy on the expert and the policy respectively
            expert_logits = discriminator(expert_data)
            policy_logits = discriminator(policy_data)
            expert_preds = torch.round(torch.sigmoid(expert_logits))
            policy_preds = torch.round(torch.sigmoid(policy_logits))
            expert_acc = (expert_preds == expert_labels).float().mean()
            policy_acc = (policy_preds == policy_labels).float().mean()
            return expert_acc, policy_acc
    else:
        return None, None

def load_human_expert_data(config, vecenv):
    """Load expert demonstrations for GAIL training. This could be bugged!"""
    
    save_path = f"irl/data/puffer_{config.train.seed}_{config.environment.max_controlled_agents}"
    trajectory_file = f"{save_path}/trajectory_0.npz"
    global_file = f"{save_path}/global/global_trajectory_0.npz"
    
    # Check if we should remake data or if data doesn't exist
    remake_data = getattr(config, 'expertdata', {}).get('remake', False)
    data_exists = os.path.exists(trajectory_file) and os.path.exists(global_file)
    
    if remake_data or not data_exists:
        if remake_data:
            print("Remaking expert demonstrations (config.expertdata.remake=True)...")
        else:
            print("Expert data not found. Generating expert demonstrations...")
            
        save_trajectory(
            env=vecenv.env,
            save_path=save_path,
            save_index=0,
            action_space_type="continuous",
            use_action_indices=False,
            save_visualization=False,
            render_index=[0, 2],
        )
    else:
        print(f"Loading existing expert data from {save_path}...")    

    # Load expert data 
    expert_data = np.load(trajectory_file)
    expert_obs = expert_data["obs"]
    collision = expert_data["veh_collision"]
    off_road = expert_data["off_road"]
    goal_achieved = expert_data["goal_achieved"]
    print(f"Off-road rate: {off_road:.3f}, Vehicle collision rate: {collision:.3f}, Goal rate: {goal_achieved:.3f}, using non-collided trajectories")

    expert_obs = torch.from_numpy(expert_obs).float()
    
    # Create and return ExpertDataset object (consistent with make_expert_dataset)
    expert_dataset = ExpertDataset(expert_obs, config)
    return expert_dataset

def get_model_parameters(policy):
    """Helper function to count the number of trainable parameters."""
    params = filter(lambda p: p.requires_grad, policy.parameters())
    return sum([np.prod(p.size()) for p in params])

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return pufferlib.namespace(**config)

def make_concatenated_obs(obs):
    """Concatenate the last observation with the current observation."""
    last_obs = torch.cat([obs[:, -1, :], obs[:, -1, :]], dim=-1)
    obs = torch.cat([obs[:, :-1, :], obs[:, 1:, :]], dim=-1)
    last_obs = last_obs.unsqueeze(1)
    obs = torch.cat([obs, last_obs], dim=1)
    return obs

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

# def run_sweep(args, vecenv, expert_dataset, project="gpudrive-gail", sweep_name="1agent_ppo_expert_sweep"):
#     """Initialize a WandB sweep with hyperparameters."""
    
#     sweep_config = {
#         "method": "random",
#         "name": sweep_name,
#         "metric": {"goal": "maximize", "name": "metrics/mean_episode_reward_per_agent"},
#         "parameters": {
#             "gail": {
#                 "discriminator_lr": {"values": [1e-4, 1e-3, 5e-4]},
#                 "use_action": {"values": [True, False]},
#                 "discriminator_hidden_dim": {"values": [16, 32, 64, 128]},
#             },
#             "train": {
#                 "batch_size": {"values": [16384, 32768, 65536]},
#             }
#         },
#     }
    
#     sweep_id = wandb.sweep(sweep=sweep_config, project=project)
#     wandb.agent(sweep_id, train(args, vecenv, expert_dataset), count=100)