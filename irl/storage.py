import torch
import numpy as np
import os
from tqdm import tqdm
import yaml
from box import Box
import pufferlib
from pathlib import Path
from PIL import Image
import logging

from gpudrive.env.config import EnvConfig, SceneConfig, SelectionDiscipline
from gpudrive.env.env_puffer import PufferGPUDrive
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return pufferlib.namespace(**config)

def map_to_closest_discrete_value(grid, cont_actions):
    """
    Find the nearest value in the action grid for a given expert action.
    """
    # Calculate the absolute differences and find the indices of the minimum values
    abs_diff = torch.abs(grid.unsqueeze(0) - cont_actions.unsqueeze(-1))
    indx = torch.argmin(abs_diff, dim=-1)

    # Gather the closest values based on the indices
    closest_values = grid[indx]

    return closest_values, indx

def save_trajectory(env, save_path, action_space_type="discrete", use_action_indices=False, save_index=0, save_visualization=False, render_index=[0, 1]):
    """
    Save the trajectory, partner_mask and road_mask in the environment, distinguishing them by each scene and agent.
    
    Args:
        env (PufferGPUDrive): Initialized environment class.
    """
    obs = env.reset()
    expert_actions, _, _, _  = env.get_expert_actions() # (num_worlds, num_agents, episode_len, action_dim)
    print(f"expert_actions: {expert_actions.shape}")
    # road_mask = env.get_road_mask()
    # partner_mask = env.get_partner_mask()
    # partner_id = env.get_partner_id().unsqueeze(-1)
    device = env.device

    if action_space_type == "discrete":
        # Discretize the expert actions: map every value to the closest
        # value in the action grid.
        disc_expert_actions = expert_actions.clone()
        if env.config.dynamics_model == "delta_local":
            disc_expert_actions[:, :, :, 0], _ = map_to_closest_discrete_value(
                grid=env.dx, cont_actions=expert_actions[:, :, :, 0]
            )
            disc_expert_actions[:, :, :, 1], _ = map_to_closest_discrete_value(
                grid=env.dy, cont_actions=expert_actions[:, :, :, 1]
            )
            disc_expert_actions[:, :, :, 2], _ = map_to_closest_discrete_value(
                grid=env.dyaw, cont_actions=expert_actions[:, :, :, 2]
            )
        else:
            # Acceleration
            disc_expert_actions[:, :, :, 0], _ = map_to_closest_discrete_value(
                grid=env.accel_actions, cont_actions=expert_actions[:, :, :, 0]
            )
            # Steering
            disc_expert_actions[:, :, :, 1], _ = map_to_closest_discrete_value(
                grid=env.steer_actions, cont_actions=expert_actions[:, :, :, 1]
            )

        if use_action_indices:  # Map action values to joint action index
            logging.info("Mapping expert actions to joint action index... \n")
            expert_action_indices = torch.zeros(
                expert_actions.shape[0],
                expert_actions.shape[1],
                expert_actions.shape[2],
                1,
                dtype=torch.int32,
            ).to(device)
            for world_idx in range(disc_expert_actions.shape[0]):
                for agent_idx in range(disc_expert_actions.shape[1]):
                    for time_idx in range(disc_expert_actions.shape[2]):
                        action_val_tuple = tuple(
                            round(x, 3)
                            for x in disc_expert_actions[
                                world_idx, agent_idx, time_idx, :
                            ].tolist()
                        )
                        if not env.config.dynamics_model == "delta_local":
                            action_val_tuple = (
                                action_val_tuple[0],
                                action_val_tuple[1],
                                0.0,
                            )

                        action_idx = env.values_to_action_key.get(
                            action_val_tuple
                        )
                        expert_action_indices[
                            world_idx, agent_idx, time_idx
                        ] = action_idx
            print(f"expert_action_indices: {expert_action_indices[0][0][0]}, {expert_action_indices[0][0][1]}, {expert_action_indices[0][0][2]}")
            print(f"expert_actions: {expert_actions[0][0][0]}, {expert_actions[0][0][1]}, {expert_actions[0][0][2]}")
            expert_actions = expert_action_indices
        else:
            # Map action values to joint action index
            expert_actions = disc_expert_actions
    elif action_space_type == "multi_discrete":
        """will be update"""
        pass
    else:
        logging.info("Using continuous expert actions... \n")
    
    cont_agent_mask = env.cont_agent_mask.to(device)  # (num_worlds, num_agents)
    alive_agent_indices = cont_agent_mask.nonzero(as_tuple=False)
    alive_agent_num = env.cont_agent_mask.sum().item()
    print("alive_agent_num : ", alive_agent_num)
    
    expert_trajectory_lst = torch.zeros((alive_agent_num, env.episode_len, obs.shape[-1]), device=device)
    expert_actions_lst = torch.zeros((alive_agent_num, env.episode_len, 3), device=device)
    expert_dead_mask_lst = torch.ones((alive_agent_num, env.episode_len), device=device, dtype=torch.bool)
    # expert_partner_mask_lst = torch.full((alive_agent_num, env.episode_len, 127), 2, device=device, dtype=torch.long)
    # expert_road_mask_lst = torch.ones((alive_agent_num, env.episode_len, 200), device=device, dtype=torch.bool)
    expert_global_pos_lst = torch.zeros((alive_agent_num, env.episode_len, 2), device=device) # global pos (2)
    expert_global_rot_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device) # global actions (1)
    
    # Initialize dead agent mask
    agent_info = (
            env.sim.absolute_self_observation_tensor()
            .to_torch()
            .to(device)
        )
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device) # (num_worlds, num_agents)
    # road_mask = env.get_road_mask()
    goal_achieved = 0
    off_road = 0
    veh_collision = 0

    # For visualization
    frames = {f"world_{i}": [] for i in range(len(render_index))}  

    for time_step in tqdm(range(env.episode_len)):
        for idx, (world_idx, agent_idx) in enumerate(alive_agent_indices):
            if not dead_agent_mask[world_idx, agent_idx]:
                expert_trajectory_lst[idx][time_step] = obs[world_idx, agent_idx]
                expert_actions_lst[idx][time_step] = expert_actions[world_idx, agent_idx, time_step]
                # expert_partner_mask_lst[idx][time_step] = partner_mask[world_idx, agent_idx]
                # expert_road_mask_lst[idx][time_step] = road_mask[world_idx, agent_idx]
                expert_global_pos_lst[idx, time_step] = agent_info[world_idx, agent_idx, 0:2]
                expert_global_rot_lst[idx, time_step] = agent_info[world_idx, agent_idx, 7:8]
            expert_dead_mask_lst[idx][time_step] = dead_agent_mask[world_idx, agent_idx]

        # env.step() -> gather next obs
        env.step_dynamics(expert_actions[:, :, time_step, :])
        dones = env.get_dones().to(device)
        
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        obs = env.get_obs() 
        # road_mask = env.get_road_mask()
        # partner_mask = env.get_partner_mask()
        # partner_id = env.get_partner_id().unsqueeze(-1)
        agent_info = (
        env.sim.absolute_self_observation_tensor()
        .to_torch()
        .to(device)
        )
        infos = env.get_infos()
        
        goal_achieved += infos.goal_achieved[cont_agent_mask]
        off_road += infos.off_road[cont_agent_mask]
        veh_collision += infos.collided[cont_agent_mask]
        goal_achieved = torch.clamp(goal_achieved, max=1.0)
        off_road = torch.clamp(off_road, max=1.0)
        veh_collision = torch.clamp(veh_collision, max=1.0)

        # Visualize
        if save_visualization:
            env_indices = render_index
            if env_indices:  # Only render if there are environments to render
                figs = env.vis.plot_simulator_state(
                    env_indices=env_indices,
                    time_steps=[time_step] * len(env_indices),
                    zoom_radius=100,
                )
                for i, env_id in enumerate(env_indices):
                    frames[f"world_{env_id}"].append(img_from_fig(figs[i]))

        if (dead_agent_mask == True).all():
            goal_rate = goal_achieved.sum().float() / cont_agent_mask.sum().float()
            off_road_rate = off_road.sum().float() / cont_agent_mask.sum().float()
            veh_coll_rate = veh_collision.sum().float() / cont_agent_mask.sum().float()
            collision = (veh_collision + off_road > 0)
            print(f'Offroad {off_road_rate} VehCol {veh_coll_rate} Goal {goal_rate}')
            print(f'Save number w/o collision {len(expert_trajectory_lst[~collision])} / {len(expert_trajectory_lst)}')
            break
    
    expert_trajectory_lst = expert_trajectory_lst[~collision].to('cpu')
    expert_actions_lst = expert_actions_lst[~collision].to('cpu')
    expert_dead_mask_lst = expert_dead_mask_lst[~collision].to('cpu')
    # expert_partner_mask_lst = expert_partner_mask_lst[~collision].to('cpu')
    # expert_road_mask_lst = expert_road_mask_lst[~collision].to('cpu')
    # global pos
    expert_global_pos_lst = expert_global_pos_lst[~collision].to('cpu')
    expert_global_rot_lst = expert_global_rot_lst[~collision].to('cpu')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + '/global', exist_ok=True)
    np.savez_compressed(f"{save_path}/trajectory_{save_index}.npz", 
                        obs=expert_trajectory_lst,
                        actions=expert_actions_lst,
                        dead_mask=expert_dead_mask_lst,
                        off_road=off_road_rate.cpu().numpy(),
                        veh_collision=veh_coll_rate.cpu().numpy(),
                        goal_achieved=goal_rate.cpu().numpy(),
                        # partner_mask=expert_partner_mask_lst,
                        # road_mask=expert_road_mask_lst
                        )
    np.savez_compressed(f"{save_path}/global/global_trajectory_{save_index}.npz", 
                        ego_global_pos=expert_global_pos_lst,
                        ego_global_rot=expert_global_rot_lst)

    # Save visualization
    if save_visualization:
        for env_id, env_frames in frames.items():
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
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--save_path', type=str, default='irl/data/full_version/processed')
    parser.add_argument('--dataset', type=str, default='training', choices=['training', 'validation', 'testing'],)
    parser.add_argument('--function', type=str, default='save_trajectory', 
                        choices=['save_trajectory'])
    args = parser.parse_args()

    torch.set_printoptions(precision=3, sci_mode=False)
    save_path = os.path.join(args.save_path, f'{args.dataset}_subset_v2')
    print()
    print("save_path : ", save_path)
    print("dataset : ", args.dataset)
    print("function : ", args.function)

    # Load configuration
    config = load_config(args.config)
    
    # Set device
    config.train.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {config.train.device}")

    if config["train"]["resample_scenes"]:
        dataset_size = config["train"]["resample_dataset_size"]
    else:
        dataset_size = config["environment"]["k_unique_scenes"]

    # Create data loader
    train_loader = SceneDataLoader(
        root=config.data_dir,
        batch_size=config.environment.num_worlds,
        dataset_size=dataset_size,
        # if config.train.resample_scenes
        # else config.environment.k_unique_scenes,
        sample_with_replacement=config.train.sample_with_replacement,
        shuffle=config.train.shuffle_dataset,
        seed=config.train.seed if hasattr(config.train, 'seed') else 42,
    )

    # Make environment
    vecenv = PufferGPUDrive(
        data_loader=train_loader,
        **config.environment,
        **config.train,
    )

    print('Launch Env')
    num_iter = int(dataset_size // config.environment.num_worlds)
    for i in tqdm(range(num_iter)):
        print(vecenv.env.data_batch)
        if args.function == 'save_trajectory':
            sv = True if i == 0 or i == num_iter - 1 else False
            save_trajectory(vecenv.env, save_path, i * config.environment.num_worlds, save_visualization=sv)
        else:
            raise ValueError("Invalid function name")
        if i != num_iter - 1:
            vecenv.env.swap_data_batch()
    vecenv.close()
    del vecenv
