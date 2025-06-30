import copy
from typing import List, Union
import torch
from torch import nn
from torch.distributions.utils import logits_to_probs
from torch.distributions import Categorical, Normal
import pufferlib.models
from gpudrive.env import constants
from huggingface_hub import PyTorchModelHubMixin
from box import Box

import madrona_gpudrive

TOP_K_ROAD_POINTS = madrona_gpudrive.kMaxAgentMapObservationsCount


def log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)


def entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)


def sample_logits(
    logits: Union[torch.Tensor, List[torch.Tensor]],
    action=None,
    deterministic=False,
):
    """Sample logits: Supports deterministic sampling."""

    normalized_logits = [logits - logits.logsumexp(dim=-1, keepdim=True)]
    logits = [logits]

    if action is None:
        if deterministic:
            # Select the action with the maximum probability
            action = torch.stack([l.argmax(dim=-1) for l in logits])
        else:
            # Sample actions stochastically from the logits
            action = torch.stack(
                [
                    torch.multinomial(logits_to_probs(l), 1).squeeze()
                    for l in logits
                ]
            )
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T

    assert len(logits) == len(action)

    logprob = torch.stack(
        [log_prob(l, a) for l, a in zip(normalized_logits, action)]
    ).T.sum(1)

    logits_entropy = torch.stack(
        [entropy(l) for l in normalized_logits]
    ).T.sum(1)

    return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0)

# def sample_continuous_actions(
#     action_mean: torch.Tensor,
#     action_logstd: torch.Tensor,
#     action=None,
#     deterministic=False,
#     action_low=None,
#     action_high=None,
# ):
#     """Sample continuous actions from a normal distribution with tanh squashing."""
#     """return action, logprob, entropy"""
#     std = torch.exp(action_logstd)
#     dist = Normal(action_mean, std)
#     if action is None:
#         if deterministic:
#             action = action_mean
#         else:
#             action = dist.sample()
#     return action, dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1)

def sample_continuous_actions(
    action_mean: torch.Tensor,
    action_logstd: torch.Tensor,
    action=None,
    deterministic=False,
    action_low=None,
    action_high=None,
):
    """Sample continuous actions from a normal distribution with tanh squashing."""
    # Ensure bounds are on the same device and broadcasted correctly
    action_low = action_low.to(action_mean.device)
    action_high = action_high.to(action_mean.device)
    
    if action is None:
        if deterministic:
            # Use the mean for deterministic action (before squashing)
            raw_action = action_mean
        else:
            # Sample from normal distribution (before squashing)
            std = torch.exp(action_logstd)
            dist = Normal(action_mean, std)
            raw_action = dist.sample()
    else:
        # Inverse tanh to get raw action from squashed action
        # Normalize action to [-1, 1] range
        action_normalized = (action - action_low) / (action_high - action_low) * 2 - 1
        # Clamp to prevent numerical issues with atanh
        action_normalized = torch.clamp(action_normalized, -0.999, 0.999)
        raw_action = torch.atanh(action_normalized)
    
    # Squash raw action to valid range using tanh
    action_normalized = torch.tanh(raw_action)  # [-1, 1]
    action = action_low + (action_normalized + 1) / 2 * (action_high - action_low)  # [action_low, action_high]
    
    # Calculate log probability with Jacobian correction for tanh squashing
    std = torch.exp(action_logstd)
    dist = Normal(action_mean, std)

    # print(f"raw_action: {raw_action[:10]}")
    # print(f"action normalized: {action[:10]}")
    
    # Log prob of raw action
    raw_logprob = dist.log_prob(raw_action).sum(dim=-1)
    
    # Jacobian correction for tanh squashing: log|d tanh(x)/dx| = log(1 - tanh²(x))
    # We need to subtract this because we're changing variables from raw_action to action
    jacobian_correction = torch.log(1 - action_normalized.pow(2) + 1e-6).sum(dim=-1)
    
    # Final log probability
    logprob = raw_logprob - jacobian_correction
    
    # Calculate entropy (of the raw distribution, since that's what we're sampling from)
    entropy = dist.entropy().sum(dim=-1)
    
    return action, logprob, entropy


class NeuralNet(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/Emerge-Lab/gpudrive",
    docs_url="https://arxiv.org/abs/2502.14706",
    tags=["ffn"],
):
    def __init__(
        self,
        action_dim=91,  # Default: 7 * 13
        input_dim=64,
        hidden_dim=128,
        dropout=0.00,
        act_func="tanh",
        max_controlled_agents=64,
        obs_dim=2984,  # Size of the flattened observation vector (hardcoded)
        config=None,  # Optional config
        continuous_actions=False,  # New parameter to specify action type
        using_shared_embedding=True,  # Whether actor and critic share embeddings
        action_low=None,  # Default action bounds for continuous actions
        action_high=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.max_controlled_agents = max_controlled_agents
        self.max_observable_agents = max_controlled_agents - 1
        self.obs_dim = obs_dim
        self.num_modes = 3  # Ego, partner, road graph
        self.dropout = dropout
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.GELU()
        self.continuous_actions = continuous_actions
        self.using_shared_embedding = using_shared_embedding
        
        # Handle action bounds for continuous actions
        if continuous_actions:
            if action_low is not None and action_high is not None:
                self.action_low = torch.tensor(action_low, dtype=torch.float32)
                self.action_high = torch.tensor(action_high, dtype=torch.float32)
            else:
                # Default bounds if not provided
                raise ValueError("Action bounds must be provided for continuous actions")
        else:
            self.action_low = None
            self.action_high = None

        # Indices for unpacking the observation
        self.ego_state_idx = constants.EGO_FEAT_DIM
        self.partner_obs_idx = (
            constants.PARTNER_FEAT_DIM * self.max_controlled_agents
        )
        if config is not None:
            self.config = Box(config)
            if "reward_type" in self.config:
                if self.config.reward_type == "reward_conditioned":
                    # Agents know their "type", consisting of three weights
                    # that determine the reward (collision, goal, off-road)
                    self.ego_state_idx += 3
                    self.partner_obs_idx += 3

            self.vbd_in_obs = self.config.vbd_in_obs

        # Calculate the VBD predictions size: 91 timesteps * 5 features = 455
        self.vbd_size = 91 * 5

        if self.using_shared_embedding:
            # Shared embeddings for both actor and critic
            self.ego_embed = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Linear(self.ego_state_idx, input_dim)
                ),
                nn.LayerNorm(input_dim),
                self.act_func,
                nn.Dropout(self.dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
            )

            self.partner_embed = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Linear(constants.PARTNER_FEAT_DIM, input_dim)
                ),
                nn.LayerNorm(input_dim),
                self.act_func,
                nn.Dropout(self.dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
            )

            self.road_map_embed = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Linear(constants.ROAD_GRAPH_FEAT_DIM, input_dim)
                ),
                nn.LayerNorm(input_dim),
                self.act_func,
                nn.Dropout(self.dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
            )

            if self.vbd_in_obs:
                self.vbd_embed = nn.Sequential(
                    pufferlib.pytorch.layer_init(
                        nn.Linear(self.vbd_size, input_dim)
                    ),
                    nn.LayerNorm(input_dim),
                    self.act_func,
                    nn.Dropout(self.dropout),
                    pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
                )

            self.shared_embed = nn.Sequential(
                nn.Linear(self.input_dim * self.num_modes, self.hidden_dim),
                nn.Dropout(self.dropout),
            )
        else:
            # Separate embeddings for actor and critic
            # Actor embeddings
            self.actor_ego_embed = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Linear(self.ego_state_idx, input_dim)
                ),
                nn.LayerNorm(input_dim),
                self.act_func,
                nn.Dropout(self.dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
            )

            self.actor_partner_embed = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Linear(constants.PARTNER_FEAT_DIM, input_dim)
                ),
                nn.LayerNorm(input_dim),
                self.act_func,
                nn.Dropout(self.dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
            )

            self.actor_road_map_embed = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Linear(constants.ROAD_GRAPH_FEAT_DIM, input_dim)
                ),
                nn.LayerNorm(input_dim),
                self.act_func,
                nn.Dropout(self.dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
            )

            if self.vbd_in_obs:
                self.actor_vbd_embed = nn.Sequential(
                    pufferlib.pytorch.layer_init(
                        nn.Linear(self.vbd_size, input_dim)
                    ),
                    nn.LayerNorm(input_dim),
                    self.act_func,
                    nn.Dropout(self.dropout),
                    pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
                )

            self.actor_embed = nn.Sequential(
                nn.Linear(self.input_dim * self.num_modes, self.hidden_dim),
                nn.Dropout(self.dropout),
            )

            # Critic embeddings
            self.critic_ego_embed = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Linear(self.ego_state_idx, input_dim)
                ),
                nn.LayerNorm(input_dim),
                self.act_func,
                nn.Dropout(self.dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
            )

            self.critic_partner_embed = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Linear(constants.PARTNER_FEAT_DIM, input_dim)
                ),
                nn.LayerNorm(input_dim),
                self.act_func,
                nn.Dropout(self.dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
            )

            self.critic_road_map_embed = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Linear(constants.ROAD_GRAPH_FEAT_DIM, input_dim)
                ),
                nn.LayerNorm(input_dim),
                self.act_func,
                nn.Dropout(self.dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
            )

            if self.vbd_in_obs:
                self.critic_vbd_embed = nn.Sequential(
                    pufferlib.pytorch.layer_init(
                        nn.Linear(self.vbd_size, input_dim)
                    ),
                    nn.LayerNorm(input_dim),
                    self.act_func,
                    nn.Dropout(self.dropout),
                    pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
                )

            self.critic_embed = nn.Sequential(
                nn.Linear(self.input_dim * self.num_modes, self.hidden_dim),
                nn.Dropout(self.dropout),
            )

        if self.continuous_actions:
            # For continuous actions, output mean and log_std
            self.actor_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_dim, action_dim), std=0.01
            )
            # Initialize log_std as a parameter (independent of state)
            self.action_logstd = nn.Parameter(torch.zeros(1, action_dim))
            embedding_type = "SHARED" if self.using_shared_embedding else "SEPARATE"
            print(f"Neural Network: Configured for CONTINUOUS actions (mean + log_std outputs) with {embedding_type} embeddings")
        else:
            # For discrete actions, output logits
            self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_dim, action_dim), std=0.01
            )
            embedding_type = "SHARED" if self.using_shared_embedding else "SEPARATE"
            print(f"Neural Network: Configured for DISCRETE actions (logits output) with {embedding_type} embeddings")
            
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_dim, 1), std=1
        )

    def encode_observations(self, observation):
        """Encode observations using shared embeddings (when using_shared_embedding=True)"""
        if not self.using_shared_embedding:
            raise ValueError("encode_observations should only be used with shared embeddings. Use encode_observations_actor/critic instead.")

        if self.vbd_in_obs:
            (
                ego_state,
                road_objects,
                road_graph,
                vbd_predictions,
            ) = self.unpack_obs(observation)
        else:
            ego_state, road_objects, road_graph = self.unpack_obs(observation)

        # Embed the ego state
        ego_embed = self.ego_embed(ego_state)

        if self.vbd_in_obs:
            vbd_embed = self.vbd_embed(vbd_predictions)
            # Concatenate the VBD predictions with the ego state embedding
            ego_embed = torch.cat([ego_embed, vbd_embed], dim=1)

        # Max pool
        partner_embed, _ = self.partner_embed(road_objects).max(dim=1)
        road_map_embed, _ = self.road_map_embed(road_graph).max(dim=1)

        # Concatenate the embeddings
        embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)

        return self.shared_embed(embed)

    def encode_observations_actor(self, observation):
        """Encode observations for the actor network"""
        if self.vbd_in_obs:
            (
                ego_state,
                road_objects,
                road_graph,
                vbd_predictions,
            ) = self.unpack_obs(observation)
        else:
            ego_state, road_objects, road_graph = self.unpack_obs(observation)

        # Embed the ego state
        ego_embed = self.actor_ego_embed(ego_state)

        if self.vbd_in_obs:
            vbd_embed = self.actor_vbd_embed(vbd_predictions)
            # Concatenate the VBD predictions with the ego state embedding
            ego_embed = torch.cat([ego_embed, vbd_embed], dim=1)

        # Max pool
        partner_embed, _ = self.actor_partner_embed(road_objects).max(dim=1)
        road_map_embed, _ = self.actor_road_map_embed(road_graph).max(dim=1)

        # Concatenate the embeddings
        embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)

        return self.actor_embed(embed)

    def encode_observations_critic(self, observation):
        """Encode observations for the critic network"""
        if self.vbd_in_obs:
            (
                ego_state,
                road_objects,
                road_graph,
                vbd_predictions,
            ) = self.unpack_obs(observation)
        else:
            ego_state, road_objects, road_graph = self.unpack_obs(observation)

        # Embed the ego state
        ego_embed = self.critic_ego_embed(ego_state)

        if self.vbd_in_obs:
            vbd_embed = self.critic_vbd_embed(vbd_predictions)
            # Concatenate the VBD predictions with the ego state embedding
            ego_embed = torch.cat([ego_embed, vbd_embed], dim=1)

        # Max pool
        partner_embed, _ = self.critic_partner_embed(road_objects).max(dim=1)
        road_map_embed, _ = self.critic_road_map_embed(road_graph).max(dim=1)

        # Concatenate the embeddings
        embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)

        return self.critic_embed(embed)

    def forward(self, obs, action=None, deterministic=False):

        if self.using_shared_embedding:
            # Use shared encodings for both actor and critic
            hidden = self.encode_observations(obs)
            
            # Decode the actions
            value = self.critic(hidden)
            
            if self.continuous_actions:
                # For continuous actions
                action_mean = self.actor_mean(hidden)
                action_logstd = self.action_logstd.expand_as(action_mean)
                
                # Move action bounds to same device as action_mean
                action_low = self.action_low.to(action_mean.device)
                action_high = self.action_high.to(action_mean.device)
                
                action, logprob, entropy = sample_continuous_actions(
                    action_mean, action_logstd, action, deterministic,
                    action_low, action_high
                )
            else:
                # For discrete actions
                logits = self.actor(hidden)
                action, logprob, entropy = sample_logits(logits, action, deterministic)
        else:
            # Use separate encodings for actor and critic
            actor_hidden = self.encode_observations_actor(obs)
            critic_hidden = self.encode_observations_critic(obs)
            
            # Decode the actions using actor encoding
            value = self.critic(critic_hidden)
            
            if self.continuous_actions:
                # For continuous actions
                action_mean = self.actor_mean(actor_hidden)
                action_logstd = self.action_logstd.expand_as(action_mean)
                
                # Move action bounds to same device as action_mean
                action_low = self.action_low.to(action_mean.device)
                action_high = self.action_high.to(action_mean.device)
                
                action, logprob, entropy = sample_continuous_actions(
                    action_mean, action_logstd, action, deterministic, action_low, action_high
                )
            else:
                # For discrete actions
                logits = self.actor(actor_hidden)
                action, logprob, entropy = sample_logits(logits, action, deterministic)

        return action, logprob, entropy, value

    def unpack_obs(self, obs_flat):
        """
        Unpack the flattened observation into the ego state, visible simulator state.

        Args:
            obs_flat (torch.Tensor): Flattened observation tensor of shape (batch_size, obs_dim).

        Returns:
            tuple: If vbd_in_obs is True, returns (ego_state, road_objects, road_graph, vbd_predictions).
                Otherwise, returns (ego_state, road_objects, road_graph).
        """

        # Unpack modalities
        ego_state = obs_flat[:, : self.ego_state_idx]
        partner_obs = obs_flat[:, self.ego_state_idx : self.partner_obs_idx]

        if self.vbd_in_obs:
            # Extract the VBD predictions (last 455 elements)
            vbd_predictions = obs_flat[:, -self.vbd_size :]

            # The rest (excluding ego_state and partner_obs) is the road graph
            roadgraph_obs = obs_flat[:, self.partner_obs_idx : -self.vbd_size]
        else:
            # Without VBD, all remaining elements are road graph observations
            roadgraph_obs = obs_flat[:, self.partner_obs_idx :]

        road_objects = partner_obs.view(
            -1, self.max_observable_agents, constants.PARTNER_FEAT_DIM
        )
        road_graph = roadgraph_obs.view(
            -1, TOP_K_ROAD_POINTS, constants.ROAD_GRAPH_FEAT_DIM
        )

        if self.vbd_in_obs:
            return ego_state, road_objects, road_graph, vbd_predictions
        else:
            return ego_state, road_objects, road_graph
