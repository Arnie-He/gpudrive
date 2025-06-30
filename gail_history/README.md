# GAIL (Generative Adversarial Imitation Learning) for GPUDrive

This implementation provides a clean interface for training GAIL on the GPUDrive environment. GAIL consists of:

1. **Generator**: A PPO policy that learns to imitate expert behavior
2. **Discriminator**: A neural network that distinguishes between expert and policy trajectories
3. **Alternating Training**: The discriminator and generator are trained in an adversarial manner

## Architecture Overview

```
Expert Data (from Waymo) → Discriminator ← Policy Trajectories
                              ↓
                         Reward Signal
                              ↓
                         PPO Policy (Generator)
```

## Key Components

### Discriminator
- Takes state-action pairs as input
- Outputs probability that the trajectory is from an expert
- Trained with binary classification loss (expert=1, policy=0)

### Generator (PPO Policy)
- Uses the existing PPO implementation from PufferLib
- Receives rewards from the discriminator instead of environment rewards
- GAIL reward: `-log(1 - D(s,a))` where D(s,a) is discriminator output

### Expert Data
- Generated from Waymo Open Dataset using `baselines/imitation_data_generation.py`
- Automatically extracted during training initialization
- State-action pairs from expert demonstrations

## Usage

### Multi-Agent GAIL Training

```bash
# Train GAIL with default parameters
python irl/gail.py run irl/config/gail_config.yaml --use-gail

# Train regular PPO for comparison
python irl/gail.py run irl/config/gail_config.yaml
```

### Single-Agent GAIL Training

For scenarios where you want each environment to have exactly one controlled agent:

```bash
# Basic single-agent GAIL
python irl/single_agent_gail.py run irl/config/single_agent_gail_config.yaml

# Limit to 32 single-agent environments
python irl/single_agent_gail.py run irl/config/single_agent_gail_config.yaml \
    --max-single-agent-envs 32 \
    --total-timesteps 1000000

# Small-scale testing (8 agents, 100k timesteps)
python irl/single_agent_gail.py run irl/config/single_agent_gail_config.yaml \
    --max-single-agent-envs 8 \
    --total-timesteps 100000
```

#### Single-Agent vs Multi-Agent

**Single-Agent GAIL** (`irl/single_agent_gail.py`):
- Each environment has exactly **one controlled agent**
- All other agents follow original Waymo trajectories  
- Creates many individual single-agent environments from multi-agent scenarios
- Better for learning individual agent behavior
- More diverse expert data (different agent positions/scenarios)
- Example: 64 agents in 2 worlds → 128 single-agent environments

**Multi-Agent GAIL** (`irl/gail.py`):
- Multiple agents controlled per environment
- Learns coordinated multi-agent behavior
- Standard GAIL setup for multi-agent systems

### Custom GAIL Parameters

```bash
python irl/gail.py run irl/config/gail_config.yaml \
    --use-gail \
    --discriminator-lr 1e-4 \
    --discriminator-hidden-dim 512 \
    --discriminator-epochs 10 \
    --discriminator-update-freq 2 \
    --policy-weight-decay 1e-4 \
    --discriminator-weight-decay 1e-4 \
    --total-timesteps 50000000
```

### Environment Configuration

```bash
python irl/gail.py run irl/config/gail_config.yaml \
    --use-gail \
    --num-worlds 256 \
    --k-unique-scenes 50 \
    --collision-weight 2.0 \
    --goal-achieved-weight 1.5
```

## Configuration

The GAIL-specific parameters in the config file:

```yaml
train:
  # GAIL-specific parameters
  use_gail: true                    # Enable GAIL training
  discriminator_lr: 3e-4           # Discriminator learning rate
  discriminator_hidden_dim: 256    # Hidden layer size
  discriminator_dropout: 0.1       # Dropout rate
  discriminator_batch_size: 1024   # Batch size for discriminator
  discriminator_epochs: 5          # Epochs per discriminator update
  discriminator_update_freq: 1     # Update frequency (every N policy updates)
  policy_buffer_size: 50000        # Size of policy trajectory buffer
  min_policy_data: 1000           # Min policy data before training discriminator
  
  # L2 Regularization
  policy_weight_decay: 1e-4        # L2 regularization for policy network
  value_weight_decay: 1e-4         # L2 regularization for value network  
  discriminator_weight_decay: 1e-4 # L2 regularization for discriminator
```

## Key Features

1. **Automatic Expert Data Generation**: Expert trajectories are automatically extracted from the Waymo dataset
2. **Flexible Architecture**: Discriminator architecture can be easily customized
3. **Integrated Logging**: WandB logging for both discriminator and policy metrics
4. **Buffer Management**: Efficient storage and sampling of policy trajectories
5. **Device Handling**: Automatic GPU/CPU device management
6. **L2 Regularization**: Built-in support for weight decay on policy, value, and discriminator networks

## Monitoring Training

The implementation logs several key metrics to WandB:

- `discriminator/loss`: Binary classification loss
- `discriminator/expert_accuracy`: Accuracy on expert data
- `discriminator/policy_accuracy`: Accuracy on policy data
- Standard PPO metrics (policy loss, value loss, etc.)

## Implementation Details

### Discriminator Architecture
- Multi-layer perceptron with LayerNorm and dropout
- Takes concatenated state-action pairs as input
- Outputs single logit for binary classification

### Reward Computation
- GAIL reward: `-log(1 - D(s,a))`
- Encourages policy to generate trajectories that fool the discriminator
- Replaces environment rewards during PPO training

### Training Alternation
- Policy rollout using current discriminator rewards
- Collect policy trajectories in buffer
- Train discriminator on expert vs policy data
- Update policy using PPO with discriminator rewards

## Troubleshooting

### Common Issues

1. **Discriminator Overfitting**: If discriminator becomes too good too quickly
   - Reduce `discriminator_lr`
   - Increase `discriminator_dropout`
   - Reduce `discriminator_epochs`

2. **Slow Policy Learning**: If policy doesn't improve
   - Increase `policy_buffer_size`
   - Reduce `discriminator_update_freq`
   - Check expert data quality

3. **Memory Issues**: If running out of GPU memory
   - Reduce `discriminator_batch_size`
   - Reduce `num_worlds`
   - Reduce `policy_buffer_size`

### Performance Tips

- Start with fewer environments (`num_worlds=128`) for debugging
- Use smaller discriminator (`discriminator_hidden_dim=128`) initially
- Monitor discriminator accuracies (should be ~50-70% for both expert and policy) 