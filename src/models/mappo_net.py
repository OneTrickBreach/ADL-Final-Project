"""
MAPPONet — Shared-parameter Actor-Critic for Multi-Agent PPO.

Architecture:
  Shared MLP backbone → Policy head (action logits) + Value head (state value).
  Value head also outputs reward decomposition estimates (aggression, preservation)
  for TensorBoard interpretability logging.

Designed for KAZ vector observations of shape (N, obs_dim) where obs_dim = 27*5 = 135.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class MAPPONet(nn.Module):
    """Shared-parameter actor-critic network for MAPPO."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
        )

        # Value head (critic) — outputs scalar state-value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Reward decomposition heads for interpretability logging
        # These predict the *component* values (aggression, preservation)
        self.aggression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.preservation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization (standard for PPO)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        # Smaller init for policy output (exploration-friendly)
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        # Smaller init for value output
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (batch, obs_dim) float tensor on CUDA.

        Returns:
            dict with keys:
                "logits":       (batch, act_dim) — raw action logits
                "value":        (batch, 1)       — state-value estimate
                "reward_decomposition": {
                    "aggression":   (batch, 1),
                    "preservation": (batch, 1),
                }
        """
        features = self.backbone(obs)
        logits = self.policy_head(features)
        value = self.value_head(features)
        reward_decomposition = {
            "aggression": self.aggression_head(features),
            "preservation": self.preservation_head(features),
        }
        return {
            "logits": logits,
            "value": value,
            "reward_decomposition": reward_decomposition,
        }

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        """
        Convenience method for PPO rollout/update.

        Args:
            obs:    (batch, obs_dim)
            action: (batch,) optional — if None, sample from policy.

        Returns:
            action, log_prob, entropy, value, reward_decomposition
        """
        out = self.forward(obs)
        dist = Categorical(logits=out["logits"])

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = out["value"].squeeze(-1)
        reward_decomposition = {
            k: v.squeeze(-1) for k, v in out["reward_decomposition"].items()
        }

        return action, log_prob, entropy, value, reward_decomposition
