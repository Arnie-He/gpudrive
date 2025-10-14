import os
import argparse
from typing import Optional
from typing_extensions import Annotated
import yaml
from datetime import datetime
import torch
import numpy as np
import wandb
from box import Box
from huggingface_hub import ModelCard

from gpudrive.integrations.puffer import ppo
from gpudrive.env.env_puffer import PufferGPUDrive, GPUDriveTorchEnv

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
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.config import EnvConfig
import dataclasses

from utils import endless_iter
from pathlib import Path
from PIL import Image

import typer
from typer import Typer
from tqdm import tqdm

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return pufferlib.namespace(**config)

def generate_ppo_expert_traj(config, seed, visualize_ppo_expert_traj=False, render_num_envs=5, maximum_episode=10, save_path=None):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vecenv = PufferGPUDrive(
        data_loader=train_loader,
        **config.environment,
        **config.train,
    )
    vecenv.async_reset(seed=seed)

    sim_agent = NeuralNet.from_pretrained(config.gail.pretrained_model_name).to(device)
    # print("sim_agent action dim: ", sim_agent.action_dim)

    card = ModelCard.load(config.gail.pretrained_model_name)

    obs_shape = vecenv.single_observation_space.shape
    print(f"obs_shape: {obs_shape}")
    control_mask = vecenv.env.cont_agent_mask
    # print(f"next_obs shape: {next_obs.shape}")

    num_envs = config.environment.num_worlds
    max_agents = control_mask.shape[1]
    # Count controlled agents for proper tensor sizing
    num_controlled_agents = control_mask.sum().item()

    frames = {f"env_{i}_firstepisode": [] for i in range(num_envs)} | {f"env_{i}_lastepisode": [] for i in range(num_envs)}
    
    # Pre-allocate tensors on GPU for better performance
    maximum_databatch_size = maximum_episode * vecenv.env.episode_len
    max_steps = min(config.train.batch_size // num_controlled_agents, maximum_databatch_size)
    
    # Pre-allocate GPU tensors to avoid repeated allocations and CPU transfers
    observations = torch.zeros((max_steps, num_controlled_agents, obs_shape[0]), 
                              device=device, dtype=torch.float32)
    actions = torch.zeros((max_steps, num_controlled_agents), 
                         device=device, dtype=torch.int64)
    last_episode_start = ((max_steps // vecenv.env.episode_len) - 1) * vecenv.env.episode_len
    
    print(f"max_steps: {max_steps}")
    print(f"vecenv.env.episode_len: {vecenv.env.episode_len}")
    print(f"last_episode_start: {last_episode_start}")

    # print(f"max_steps: {max_steps}")
    for time_step in tqdm(range(max_steps), desc="Generating expert trajectory", unit="step"):
        (
            next_obs,
            reward,
            terminal,
            truncated,
            info,
            env_id,
            mask,
        ) = vecenv.recv()
        print(f"step {time_step}, info: {info}")
        # Predict actions
        action, _, _, _ = sim_agent(
            next_obs, deterministic=False
        )

        # Step
        vecenv.send(action)

        if((time_step <= vecenv.env.episode_len) and visualize_ppo_expert_traj):
            # Render    
            sim_states = vecenv.env.vis.plot_simulator_state(
                env_indices=list(range(render_num_envs)),
                time_steps=[time_step]*render_num_envs,
                zoom_radius=70,
            )
            for i in range(render_num_envs):
                frames[f"env_{i}_firstepisode"].append(img_from_fig(sim_states[i])) 
        
        # if(time_step >= last_episode_start) and visualize_ppo_expert_traj:
        #     sim_states = vecenv.env.vis.plot_simulator_state(
        #         env_indices=list(range(render_num_envs)),
        #         time_steps=[time_step]*render_num_envs,
        #         zoom_radius=70,
        #     )
        #     for i in range(render_num_envs):
        #         frames[f"env_{i}_lastepisode"].append(img_from_fig(sim_states[i])) 

        # Store data directly on GPU - no CPU conversion until the end
        observations[time_step] = next_obs
        actions[time_step] = action

    vecenv.close()

    if save_path is None:
        save_path = "irl/data/ppo_expert_traj"
    save_index = 0

    os.makedirs(save_path, exist_ok=True)
    
    # Convert to CPU and numpy only at the end for maximum efficiency
    observations_cpu = observations.cpu().numpy()
    actions_cpu = actions.cpu().numpy()

    # Flatten the arrays to desired shapes
    observations_cpu = observations_cpu.reshape(-1, observations_cpu.shape[-1]) 
    actions_cpu = actions_cpu.reshape(-1, 1)  
    # print(f"observations_cpu shape after flattening: {observations_cpu.shape}")
    # print(f"actions_cpu shape after flattening: {actions_cpu.shape}")
    
    np.savez_compressed(f"{save_path}/trajectory_{save_index}.npz", 
                        obs=observations_cpu,
                        actions=actions_cpu,
                        )

    if visualize_ppo_expert_traj:
        for env_id, env_frames in frames.items():
            # Only save GIF if we have frames to save
            if len(env_frames) > 0:
                output_path = Path(save_path) / f"{env_id}_{save_index}.gif"
                # Convert frames to PIL Images and save as GIF
                pil_frames = [Image.fromarray(frame) for frame in env_frames]
                pil_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=200,  # 200ms per frame (5 fps)
                    loop=0
                )
                print(f"Saved video to {output_path}")
            # else:
                # print(f"Skipping {env_id} - no frames collected")
    # save the trajectory 

def main(batch_size, num_agents, visualize_ppo_expert_traj):
    config = load_config("irl/config/gail_base_puffer.yaml")
    config.train.batch_size = batch_size
    config.environment.max_controlled_agents = num_agents
    generate_ppo_expert_traj(config, seed=42, visualize_ppo_expert_traj=visualize_ppo_expert_traj, render_num_envs=2, maximum_episode=10, save_path=f"irl/data/ppo_expert_traj_{config.environment.max_controlled_agents}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PPO expert trajectories")
    parser.add_argument("--batch_size", type=int, default=65536, help="Batch size for training (default: 32768)")
    parser.add_argument("--num_agents", type=int, default=64, help="Number of controlled agents (default: 3)")
    parser.add_argument("--visualize_ppo_expert_traj", type=bool, default=True, help="Visualize PPO expert trajectories (default: True)")
    
    args = parser.parse_args()
    main(batch_size=args.batch_size, num_agents=args.num_agents, visualize_ppo_expert_traj=args.visualize_ppo_expert_traj)