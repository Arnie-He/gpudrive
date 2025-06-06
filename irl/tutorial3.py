import os
from pathlib import Path
import torch
import mediapy
from PIL import Image

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader

MAX_NUM_OBJECTS = 64  # Maximum number of objects in the scene we control
NUM_WORLDS = 2  # Number of parallel environments
UNIQUE_SCENES = 2 # Number of unique scenes

device = 'cpu'

env_config = EnvConfig(
    steer_actions = torch.round(
        torch.linspace(-1.0, 1.0, 3), decimals=3),
    accel_actions = torch.round(
        torch.linspace(-3, 3, 3), decimals=3
    )
)

# Make dataloader
data_loader = SceneDataLoader(
    root="data/processed/examples", # Path to the dataset
    batch_size=NUM_WORLDS, # Batch size, you want this to be equal to the number of worlds (envs) so that every world receives a different scene
    dataset_size=UNIQUE_SCENES, # Total number of different scenes we want to use
    sample_with_replacement=False, 
    seed=42, 
    shuffle=True,   
)

# Make environment
env = GPUDriveTorchEnv(
    config=env_config,
    data_loader=data_loader,
    max_cont_agents=MAX_NUM_OBJECTS, # Maximum number of agents to control per scenario
    device=device,
)

obs = env.reset()

frames = {f"env_{i}": [] for i in range(NUM_WORLDS)}

for t in range(env_config.episode_len):
    
    # Sample random actions
    rand_action = torch.Tensor(
        [[env.action_space.sample() for _ in range(MAX_NUM_OBJECTS * NUM_WORLDS)]]
    ).reshape(NUM_WORLDS, MAX_NUM_OBJECTS)

    # Step the environment
    env.step_dynamics(rand_action)

    obs = env.get_obs()
    reward = env.get_rewards()
    done = env.get_dones()

    # Render the environment    
    if t % 5 == 0:
        imgs = env.vis.plot_simulator_state(
            env_indices=list(range(NUM_WORLDS)),
            time_steps=[t]*NUM_WORLDS,
            zoom_radius=70,
        )
    
        for i in range(NUM_WORLDS):
            frames[f"env_{i}"].append(img_from_fig(imgs[i])) 
        
    if done.all():
        break

# Create output directory if it doesn't exist
output_dir = Path("outputs/videos")
output_dir.mkdir(parents=True, exist_ok=True)
# Save each environment's video as a separate GIF file
for env_id, env_frames in frames.items():
    output_path = output_dir / f"{env_id}.gif"
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