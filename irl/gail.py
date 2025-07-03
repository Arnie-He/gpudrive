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
import numpy as np
import wandb
from box import Box

from gpudrive.integrations.puffer import ppo
from gpudrive.env.env_puffer import PufferGPUDrive

from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.dataset import SceneDataLoader
from storage import save_trajectory

import pufferlib
import pufferlib.vector
import pufferlib.cleanrl
from rich.console import Console
import torch.utils.data as thd
import torch.nn as nn
import torch.nn.functional as F

from utils import endless_iter

import typer
from typer import Typer

app = Typer()

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

def load_expert_data(config, vecenv):
    """Load expert demonstrations for GAIL training."""
    
    save_path = f"irl/data/puffer_{config.train.seed}"
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
    
    return expert_obs

class ExpertDataset:
    def __init__(self, expert_obs, config):
        self.batch_size = config.gail.data_batch_size
        # expert_obs: (num_samples, num_steps, obs_dim)
        print(f"Expert observations shape: {expert_obs.shape}, batch size: {self.batch_size}")
        
        if(config.gail.use_consecutive_obs):
            self.expert_obs = make_concatenated_obs(expert_obs)
            print(f"Concatenated consecutive observations shape: {self.expert_obs.shape}")
        else:
            self.expert_obs = expert_obs
        
        # Reshape to (num_samples*num_steps, obs_dim)
        self.expert_obs = self.expert_obs.reshape(-1, self.expert_obs.shape[-1])
        
    #   self.expert_data_loader = thd.DataLoader(self.expert_obs, batch_size=self.batch_size, shuffle=True)
    #     # Create an endless iterator that cycles through the dataset infinitely
    #     self.endless_iter = endless_iter(self.expert_data_loader)
    
    # def next_iter(self):
    #     """Get the next batch from the endless iterator."""
    #     return next(self.endless_iter)
    
    def next_iter(self):
        """Get a randomly shuffled batch from the expert dataset."""
        # Randomly sample batch_size indices from the entire expert dataset
        total_samples = self.expert_obs.shape[0]
        random_indices = torch.randperm(total_samples)[:self.batch_size]
        return self.expert_obs[random_indices]


class Discriminator(nn.Module):
    """Discriminator network for GAIL that distinguishes expert from policy trajectories based on states only."""
    
    def __init__(self, obs_dim, hidden_dim=32, dropout=0.0, config=None):
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        assert reward_type in ["classic"], "Supported reward types: classic"
        logits = self.forward(states)
        if reward_type == "classic":
            reward = -F.logsigmoid(-self.forward(states))
        return reward

def train_discriminator(policy_data, expert_data, discriminator, optimizer, minibatch_size=None, test_accuracy=False):
    # Get the device from discriminator's parameters
    device = next(discriminator.parameters()).device
    
    policy_data = policy_data.to(device)
    expert_data = expert_data.to(device)

    expert_labels = torch.ones(expert_data.shape[0], 1, device=device)
    policy_labels = torch.zeros(policy_data.shape[0], 1, device=device)

    print(f"expert_data shape: {expert_data.shape}, policy_data shape: {policy_data.shape}")

    all_obs = torch.cat([expert_data, policy_data])
    all_labels = torch.cat([expert_labels, policy_labels])
    # shuffle data
    indices = torch.randperm(all_obs.shape[0])
    all_obs = all_obs[indices]
    all_labels = all_labels[indices]
    
    # Validate minibatch_size
    if minibatch_size is None or minibatch_size <= 0:
        minibatch_size = all_obs.shape[0]  # Use full batch if not specified
    minibatch_size = min(minibatch_size, all_obs.shape[0])  # Don't exceed available data
    
    for i in range(0, all_obs.shape[0], minibatch_size):
        minibatch_obs = all_obs[i:i+minibatch_size]
        minibatch_labels = all_labels[i:i+minibatch_size]
        optimizer.zero_grad()
        logits = discriminator(minibatch_obs)
        bce_loss = F.binary_cross_entropy_with_logits(logits, minibatch_labels)
        bce_loss.backward()
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


def train(args, vecenv):
    """
    Main training loop for the GAIL agent.
    Args:
        args: The configuration object.
        vecenv: The vectorized environment.
    Returns:
        None
    Alternates between the discriminator training and the policy training.
    """
    policy = make_agent(env=vecenv.driver_env, config=args).to(
        args.train.device
    )

    args.train.network.num_parameters = get_model_parameters(policy)
    args.train.env = args.environment.name

    args.wandb = init_wandb(args, args.train.exp_id, id=args.train.exp_id)
    args.train.__dict__.update(dict(args.wandb.config.train))

    data = ppo.create(args.train, vecenv, policy, wandb=args.wandb)

    # Initialize expert dataset
    expert_obs = load_expert_data(args, vecenv)
    expert_dataset = ExpertDataset(expert_obs, args)

    # Calculate obs_dim from actual data shape
    base_obs_dim = expert_obs.shape[-1]
    if(args.gail.use_consecutive_obs):
        obs_dim = base_obs_dim * 2
    else:
        obs_dim = base_obs_dim
    
    discriminator = Discriminator(obs_dim, args.gail.discriminator_hidden_dim, args.gail.discriminator_dropout, args)
    discriminator = discriminator.to(args.train.device)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.gail.discriminator_lr)

    step_count = 0
    while data.global_step < args.train.total_timesteps:
        try:
            ppo.evaluate(data)  # Rollout
            obs_data = data.experience.obs[:data.experience.ptr]
            expert_data = expert_dataset.next_iter()
            # print(f"obs_data shape: {obs_data.shape}, expert_data shape: {expert_data.shape}")
            if(args.gail.use_consecutive_obs):
                obs_data = make_concatenated_obs(obs_data)
                obs_data = obs_data.reshape(-1, obs_data.shape[-1])
            test_accuracy = step_count % args.gail.evaluate_discriminator_every == 0
            (expert_acc, policy_acc) = train_discriminator(obs_data, expert_data, discriminator, discriminator_optimizer, args.gail.minibatch_size, test_accuracy=test_accuracy)
   
            # Replace environment rewards with discriminator rewards
            with torch.no_grad():
                disc_rewards = discriminator.predict_reward(
                    obs_data.to(args.train.device),
                    reward_type="classic"
                )
                disc_rewards = disc_rewards.view(data.experience.rewards[:data.experience.ptr].shape)
                # print(f"disc_rewards samples: {disc_rewards[:10]}")
                # print(f"real rewards samples: {data.experience.rewards[:data.experience.ptr][:10]}")
                data.experience.rewards[:data.experience.ptr] = disc_rewards.cpu()
                # print(f"real rewards samples after: {data.experience.rewards[:data.experience.ptr][:10]}")
            ppo.train(data)

            if(test_accuracy):
                wandb.log({
                    "discriminator/expert_acc": expert_acc,
                    "discriminator/policy_acc": policy_acc,
                })
            
            step_count += 1

        except KeyboardInterrupt:   
            ppo.close(data)
            os._exit(0)
        except Exception as e:
            print(f"An error occurred: {e}")  # Log the error
            Console().print_exception()
            os._exit(1)  # Exit with a non-zero status to indicate an error

    ppo.evaluate(data)
    ppo.close(data)


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
            "gail": dict(args.gail),
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
        str, typer.Argument(help="The path to the default configuration file")
    ] = "irl/config/gail_base_puffer.yaml",
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
    # action_type: Annotated[Optional[str], typer.Option(help="Action space type; 'discrete' or 'continuous'")] = None,
    use_vbd: Annotated[Optional[bool], typer.Option(help="Use VBD model for trajectory predictions")] = False,
    vbd_model_path: Annotated[Optional[str], typer.Option(help="Path to VBD model checkpoint")] = None,
    vbd_trajectory_weight: Annotated[Optional[float], typer.Option(help="Weight for VBD trajectory deviation penalty")] = 0.1,
    vbd_in_obs: Annotated[Optional[bool], typer.Option(help="Include VBD predictions in the observation")] = False,
    init_steps: Annotated[Optional[int], typer.Option(help="Environment warmup steps")] = 0,
    # GAIL-specific options
    gail_batch_size: Annotated[Optional[int], typer.Option(help="The batch size for training")] = None,
    # Train options
    seed: Annotated[Optional[int], typer.Option(help="The seed for training")] = 42,
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
    # Wandb logging options
    project: Annotated[Optional[str], typer.Option(help="WandB project name")] = None,
    entity: Annotated[Optional[str], typer.Option(help="WandB entity name")] = None,
    group: Annotated[Optional[str], typer.Option(help="WandB group name")] = None,
    render: Annotated[Optional[int], typer.Option(help="Whether to render the environment; 0 or 1")] = None,
):
    """Run PPO training with the given configuration."""
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
        # "action_type": action_type,
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
        "gail_batch_size": gail_batch_size,
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

    if config["train"]["resample_scenes"]:
        if config["train"]["resample_scenes"]:
            dataset_size = config["train"]["resample_dataset_size"]
        config["train"][
            "exp_id"
        ] = f'{config["train"]["exp_id"]}__{cont_train}__R_{dataset_size}__{datetime_}'
    else:
        dataset_size = str(config["environment"]["k_unique_scenes"])
        config["train"][
            "exp_id"
        ] = f'{config["train"]["exp_id"]}__{cont_train}__S_{dataset_size}__{datetime_}'

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

    train(config, vecenv)


if __name__ == "__main__":

    app()