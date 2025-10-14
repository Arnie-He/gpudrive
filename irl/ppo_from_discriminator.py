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
from generate_ppo_expert_traj import generate_ppo_expert_traj

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
        
def train(args, vecenv, expert_dataset, discriminator_path=None):
    """
    Main training loop for the GAIL agent.
    Args:
        args: The configuration object.
        vecenv: The vectorized environment.
    Returns:
        None
    Alternates between the discriminator training and the policy training.
    """
    # Initialize expert dataset
    policy = make_agent(env=vecenv.driver_env, config=args).to(
        args.train.device
    )

    args.train.network.num_parameters = get_model_parameters(policy)
    args.train.env = args.environment.name

    run_name = f"{args.train.exp_id}_{args.environment.max_controlled_agents}agent(s)"
    if(args.gail.use_generated_expert):
        run_name = f"ppo_expert_{run_name}"
    else:
        run_name = f"human_expert_{run_name}"
    
    if(args.train.render):
        run_name += "_vis"
    
    # Check if wandb is already initialized (by sweep agent)
    if wandb.run is None:
        args.wandb = init_wandb(args, run_name, id=args.train.exp_id)
        args.train.__dict__.update(dict(args.wandb.config.train))
    else:
        # wandb already initialized by sweep, just use it
        args.wandb = wandb

    data = ppo.create(args.train, vecenv, policy, wandb=args.wandb)

    # Calculate obs_dim from actual data shape
    obs_dim = expert_dataset.expert_obs.shape[-1]
    
    # Initialize discriminator and optimizer
    discriminator = Discriminator(obs_dim, args.gail.discriminator_hidden_dim, args.gail.discriminator_dropout, args)
    discriminator = discriminator.to(args.train.device)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), 
        lr=args.gail.discriminator_lr,
        weight_decay=args.gail.discriminator_wd
    )
    
    # Load the discriminator if the path is provided
    loaded_policy = None
    if discriminator_path is not None:
        print(f"Loading discriminator from {discriminator_path}")
        checkpoint = torch.load(discriminator_path, map_location=args.train.device, weights_only=False)
        
        # Validate checkpoint contains required keys
        required_keys = ['discriminator', 'discriminator_optimizer']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        
        # Load discriminator state
        if 'discriminator' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator'])
            print("Loaded discriminator state dict")
        
        # Load discriminator optimizer state
        if 'discriminator_optimizer' in checkpoint:
            discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
            print("Loaded discriminator optimizer state dict")
        
        # Load policy for comparison if available
        if 'policy' in checkpoint:
            loaded_policy = make_agent(env=vecenv.driver_env, config=args).to(args.train.device)
            loaded_policy.load_state_dict(checkpoint['policy'])
            print("Loaded policy from checkpoint for comparison")
    
    step_count = 0
    while data.global_step < args.train.total_timesteps:
        try:
            ppo.evaluate(data)  # Rollout
            obs_data = data.experience.obs[:data.experience.ptr]
            # concat actions to obs_data if use_action is true
            if(args.gail.use_action):
                actions_data = data.experience.actions[:data.experience.ptr].to(args.train.device).unsqueeze(-1)
                print(f"obs_data shape: {obs_data.shape}, actions shape: {actions_data.shape}")
                obs_data = torch.cat([obs_data, actions_data], axis=-1)
            
            if(args.gail.use_full_dataset):
                expert_data = expert_dataset.expert_obs
                generator_data = obs_data
            else:
                expert_data = expert_dataset.next_iter()
                generator_data = obs_data[torch.randperm(obs_data.shape[0])[:expert_dataset.batch_size]]
            
            test_accuracy = step_count % args.gail.evaluate_discriminator_every == 0
            buffered_expert_acc, buffered_policy_acc = train_discriminator(generator_data, expert_data, discriminator, discriminator_optimizer, test_accuracy=test_accuracy)
            if buffered_expert_acc is not None:
                expert_acc = buffered_expert_acc
                policy_acc = buffered_policy_acc
   
            # Replace environment rewards with discriminator rewards
            with torch.no_grad():
                disc_rewards = discriminator.predict_reward(
                    obs_data.to(args.train.device),
                    reward_type=args.gail.reward_type
                )
                disc_rewards = disc_rewards.view(data.experience.rewards[:data.experience.ptr].shape)
                # print(f"disc_rewards samples: {disc_rewards[:10]}")
                # print(f"real rewards samples: {data.experience.rewards[:data.experience.ptr][:10]}")
                if(not(args.gail.test_vanilla_ppo)):
                    data.experience.rewards[:data.experience.ptr] = disc_rewards.cpu()
                # print(f"real rewards samples after: {data.experience.rewards[:data.experience.ptr][:10]}")
            ppo.train(data)
            
            if(test_accuracy):
                log_dict = {
                    "discriminator/expert_acc": expert_acc,
                    "discriminator/policy_acc": policy_acc,
                    "discriminator/expert_data_size": expert_data.shape[0],
                    "discriminator/policy_data_size": generator_data.shape[0],
                }

            if(step_count % args.train.save_model_every == 0):
                print(f"Saving model at step {step_count}")
                # create the save path if it doesn't exist
                os.makedirs(f"{args.train.save_model_path}/{args.train.exp_id}_{args.environment.max_controlled_agents}agent(s)", exist_ok=True)
                torch.save({
                    "discriminator": discriminator.state_dict(), 
                    "discriminator_optimizer": discriminator_optimizer.state_dict(),
                    "policy": data.uncompiled_policy.state_dict(),
                    "policy_optimizer": data.optimizer.state_dict(),
                    "step_count": step_count,
                    "discriminator_acc": (expert_acc, policy_acc),
                    }, f"{args.train.save_model_path}/{args.train.exp_id}_{args.environment.max_controlled_agents}agent(s)/model_step_{step_count}.pth")
            
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

# def run_sweep(args, vecenv, expert_dataset, project="gpudrive-gail", sweep_name="1agent_ppo_expert_sweep"):
#     """Initialize a WandB sweep with hyperparameters."""
    
    # sweep_config = {
    #     "method": "random",
    #     "name": sweep_name,
    #     "metric": {"goal": "maximize", "name": "metrics/mean_episode_reward_per_agent"},
    #     "parameters": {
    #         "gail": {
    #             "discriminator_lr": {"values": [1e-4, 1e-3, 5e-4]},
    #             "use_action": {"values": [True, False]},
    #             "discriminator_hidden_dim": {"values": [16, 32, 64, 128]},
    #         },
    #         "train": {
    #             "batch_size": {"values": [16384, 32768, 65536]},
    #         }
    #     },
    # }
    
#     sweep_id = wandb.sweep(sweep=sweep_config, project=project)
#     wandb.agent(sweep_id, train(args, vecenv, expert_dataset), count=100)


def make_expert_dataset(config, seed=42):
    save_path = f"irl/data/ppo_expert_traj_{config.environment.max_controlled_agents}"
    trajectory_file = f"{save_path}/trajectory_0.npz"
    remake_data = getattr(config, 'expertdata', {}).get('remake', False)
    data_exists = os.path.exists(trajectory_file)
    if remake_data or not data_exists:
        if remake_data:
            print("Remaking expert demonstrations (config.expertdata.remake=True)...")
        else:
            print("Expert data not found. Generating expert demonstrations...")
            
        generate_ppo_expert_traj(config, seed=config.train.seed, save_path=save_path)
    else:
        print(f"Loading existing expert data from {save_path}...") 
    
    expert_data = np.load(trajectory_file)
    expert_obs = torch.from_numpy(expert_data["obs"]).float()
    expert_actions = torch.from_numpy(expert_data["actions"]).float()
    if(config.gail.use_action):
        print(f"Using actions, obs shape: {expert_obs.shape}, actions shape: {expert_actions.shape}")
        expert_obs = torch.cat([expert_obs, expert_actions], dim=-1)
    
    expert_dataset = ExpertDataset(expert_obs, config)
    return expert_dataset

def load_human_expert_data(config, vecenv):
    """Load expert demonstrations for GAIL training."""
    
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
    max_controlled_agents: Annotated[Optional[int], typer.Option(help="Maximum number of controlled agents")] = None,
    # action_type: Annotated[Optional[str], typer.Option(help="Action space type; 'discrete' or 'continuous'")] = None,
    use_vbd: Annotated[Optional[bool], typer.Option(help="Use VBD model for trajectory predictions")] = False,
    vbd_model_path: Annotated[Optional[str], typer.Option(help="Path to VBD model checkpoint")] = None,
    vbd_trajectory_weight: Annotated[Optional[float], typer.Option(help="Weight for VBD trajectory deviation penalty")] = 0.1,
    vbd_in_obs: Annotated[Optional[bool], typer.Option(help="Include VBD predictions in the observation")] = False,
    init_steps: Annotated[Optional[int], typer.Option(help="Environment warmup steps")] = 0,    
    # GAIL-specific options
    test_vanilla_ppo: Annotated[Optional[int], typer.Option(help="Whether to test vanilla PPO; 0 or 1")] = None,
    discriminator_lr: Annotated[Optional[float], typer.Option(help="The learning rate for the discriminator")] = None,
    discriminator_hidden_dim: Annotated[Optional[int], typer.Option(help="The hidden dimension for the discriminator")] = None,
    discriminator_dropout: Annotated[Optional[float], typer.Option(help="The dropout rate for the discriminator")] = None,
    discriminator_wd: Annotated[Optional[float], typer.Option(help="The weight decay for the discriminator")] = None,
    discriminator_batch_size: Annotated[Optional[int], typer.Option(help="The batch size for training")] = None,
    use_full_dataset: Annotated[Optional[int], typer.Option(help="Whether to use the full dataset for training")] = None,
    use_action: Annotated[Optional[int], typer.Option(help="Whether to use actions for training")] = None,
    use_generated_expert: Annotated[Optional[int], typer.Option(help="Whether to use generated expert data; 0 or 1")] = None,
    reward_type: Annotated[Optional[str], typer.Option(help="The type of reward to use; 'classic', 'AIRL', or 'negativelnD'")] = None,
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
    num_minibatches: Annotated[Optional[int], typer.Option(help="The number of minibatches for training")] = None,
    gamma: Annotated[Optional[float], typer.Option(help="The discount factor for rewards")] = None,
    vf_coef: Annotated[Optional[float], typer.Option(help="Weight for vf_loss")] = None,
    weight_decay: Annotated[Optional[float], typer.Option(help="Weight decay for training")] = None,
    save_model_every: Annotated[Optional[int], typer.Option(help="The frequency to save the model")] = None,
    # Wandb logging options
    project: Annotated[Optional[str], typer.Option(help="WandB project name")] = None,
    entity: Annotated[Optional[str], typer.Option(help="WandB entity name")] = None,
    group: Annotated[Optional[str], typer.Option(help="WandB group name")] = None,
    render: Annotated[Optional[int], typer.Option(help="Whether to render the environment; 0 or 1")] = None,
    render_k_scenarios: Annotated[Optional[int], typer.Option(help="The number of scenarios to render")] = None,

    sweep: Annotated[Optional[int], typer.Option(help="Whether to run a sweep; 0 or 1")] = None,
    name: Annotated[Optional[str], typer.Option(help="The name of the run")] = None,

    discriminator_path: Annotated[Optional[str], typer.Option(help="The path to the discriminator model")] = None,
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
        "max_controlled_agents": max_controlled_agents,
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
        "num_minibatches": num_minibatches,
        "render": None if render is None else bool(render),
        "render_k_scenarios": render_k_scenarios,
        "gamma": gamma,
        "vf_coef": vf_coef,
        "weight_decay": weight_decay,
        "save_model_every": save_model_every,
    }
    config.train.update(
        {k: v for k, v in train_config.items() if v is not None}
    )

    gail_config = {
        "test_vanilla_ppo": None if test_vanilla_ppo is None else bool(test_vanilla_ppo),
        "discriminator_wd": discriminator_wd,
        "data_batch_size": discriminator_batch_size,
        "use_full_dataset": None if use_full_dataset is None else bool(use_full_dataset),
        "use_action": None if use_action is None else bool(use_action),
        "discriminator_lr": discriminator_lr,
        "discriminator_hidden_dim": discriminator_hidden_dim,
        "discriminator_dropout": discriminator_dropout,
        "use_generated_expert": None if use_generated_expert is None else bool(use_generated_expert),
        "reward_type": reward_type,
        "discriminator_path": discriminator_path,
    }
    config.gail.update(
        {k: v for k, v in gail_config.items() if v is not None}
    )

    wandb_config = {
        "project": project,
        "entity": entity,
        "group": group,
    }
    config.wandb.update(
        {k: v for k, v in wandb_config.items() if v is not None}
    )

    config.train.minibatch_size = config.train.batch_size // config.train.num_minibatches

    datetime_ = datetime.now().strftime("%m_%d_%H_%M_%S_%f")[:-3]
    datetime_ = "_" + name if name else "" + "_" + datetime_

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
    
    assert not(not(config.gail.use_generated_expert) and config.gail.use_action), "Cannot use actions with human expert data"
    if(config.gail.use_generated_expert):
        expert_dataset = make_expert_dataset(config)
    else:
        expert_dataset = load_human_expert_data(config, vecenv)
    train(config, vecenv, expert_dataset, discriminator_path)   

if __name__ == "__main__":
    app()