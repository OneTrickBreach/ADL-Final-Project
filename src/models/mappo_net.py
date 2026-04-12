"""
MAPPONet — Shared-parameter Actor-Critic for Multi-Agent PPO.

Architecture:
  Shared MLP backbone → Policy head (action logits) + Value head (state value).
  Value head also outputs reward decomposition estimates (aggression, preservation)
  for TensorBoard interpretability logging.

Designed for KAZ vector observations of shape (N, obs_dim) where obs_dim = num_entities * entity_dim.

Supported architectures:
  'mlp'           — 2-layer MLP backbone (G1–G3)
  'attention'     — entity self-attention encoder (G4)
  'attention_gru' — attention encoder + GRUCell temporal memory (G5)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class EntityAttentionEncoder(nn.Module):
    """Entity-level self-attention for structured (num_entities, entity_dim) observations.

    Reshapes a flat observation vector into entity slots, projects each entity
    to hidden_dim, applies multi-head self-attention (masking out zero-padded
    entities), and mean-pools over active entities to produce a fixed-size
    representation.
    """

    def __init__(self, entity_dim: int, num_entities: int,
                 hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.entity_dim = entity_dim
        self.num_entities = num_entities
        self.hidden_dim = hidden_dim

        # Project each entity's raw features to hidden_dim
        self.entity_embed = nn.Linear(entity_dim, hidden_dim)

        # Multi-head self-attention over entity slots
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """Args: obs_flat (batch, num_entities * entity_dim). Returns (batch, hidden_dim)."""
        batch = obs_flat.shape[0]
        entities = obs_flat.view(batch, self.num_entities, self.entity_dim)

        # Padding mask: True = IGNORE (zero-padded entity)
        padding_mask = (entities.abs().sum(dim=-1) < 1e-6)  # (batch, num_entities)

        # Embed entities
        embedded = self.entity_embed(entities)  # (batch, N, hidden_dim)

        # Self-attention with padding mask
        attended, _ = self.attention(
            embedded, embedded, embedded, key_padding_mask=padding_mask,
        )

        # Residual connection + LayerNorm
        features = self.norm(attended + embedded)

        # Mean pool over non-padded entities only
        active = (~padding_mask).unsqueeze(-1).float()       # (batch, N, 1)
        pooled = (features * active).sum(dim=1) / active.sum(dim=1).clamp(min=1)

        return pooled  # (batch, hidden_dim)


class MAPPONet(nn.Module):
    """Shared-parameter actor-critic network for MAPPO.

    Supports multiple encoder architectures:
      'mlp'           — standard 2-layer MLP backbone (G1–G3)
      'attention'     — entity self-attention encoder (G4)
      'attention_gru' — attention + GRUCell temporal memory (G5)
    """

    ARCH_CHOICES = ("mlp", "attention", "attention_gru")

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 arch: str = "mlp",
                 num_entities: int = 27, entity_dim: int = 5,
                 num_heads: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.arch = arch
        self.hidden_dim = hidden_dim

        # ── Encoder ──────────────────────────────────────────
        if arch == "mlp":
            self.backbone = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        elif arch == "attention":
            self.entity_attention = EntityAttentionEncoder(
                entity_dim, num_entities, hidden_dim, num_heads,
            )
            self.backbone = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        elif arch == "attention_gru":
            self.entity_attention = EntityAttentionEncoder(
                entity_dim, num_entities, hidden_dim, num_heads,
            )
            # GRUCell processes one timestep at a time during rollout
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
            self.backbone = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown arch '{arch}', expected one of {self.ARCH_CHOICES}")

        # ── Policy head (actor) ──────────────────────────────
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
        )

        # ── Value head (critic) ──────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Reward decomposition heads for interpretability logging
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

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None):
        """
        Args:
            obs:    (batch, obs_dim) float tensor on the model's device.
            hidden: (batch, hidden_dim) GRU hidden state — only used for
                    'attention_gru' arch; ignored (and returned as None) otherwise.

        Returns:
            dict with keys:
                "logits":       (batch, act_dim) — raw action logits
                "value":        (batch, 1)       — state-value estimate
                "hidden":       (batch, hidden_dim) new GRU state, or None
                "reward_decomposition": {
                    "aggression":   (batch, 1),
                    "preservation": (batch, 1),
                }
        """
        new_hidden = None
        if self.arch == "attention":
            attn_out = self.entity_attention(obs)
            features = self.backbone(attn_out)
        elif self.arch == "attention_gru":
            attn_out = self.entity_attention(obs)
            if hidden is None:
                hidden = torch.zeros(
                    obs.shape[0], self.hidden_dim, device=obs.device, dtype=obs.dtype
                )
            new_hidden = self.gru(attn_out, hidden)
            features = self.backbone(new_hidden)
        else:
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
            "hidden": new_hidden,
            "reward_decomposition": reward_decomposition,
        }

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor = None,
        hidden: torch.Tensor = None,
    ):
        """
        Convenience method for PPO rollout/update.

        Args:
            obs:    (batch, obs_dim)
            action: (batch,) optional — if None, sample from policy.
            hidden: (batch, hidden_dim) optional GRU hidden state.

        Returns:
            action, log_prob, entropy, value, reward_decomposition, new_hidden
            For non-GRU arches, new_hidden is None.
        """
        out = self.forward(obs, hidden=hidden)
        dist = Categorical(logits=out["logits"])

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = out["value"].squeeze(-1)
        reward_decomposition = {
            k: v.squeeze(-1) for k, v in out["reward_decomposition"].items()
        }
        new_hidden = out["hidden"]

        return action, log_prob, entropy, value, reward_decomposition, new_hidden
