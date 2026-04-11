"""
KAZWrapper — Unified wrapper for Knights-Archers-Zombies (PettingZoo).

Accepts a `game_level` argument (1–5) and toggles constraints accordingly:
  Game 1: No constraints (pass-through). Reward = kills only.
  Game 2: + Ammo restriction for archers.         (Phase 2)
  Game 3: + Stamina decay for knights.             (Phase 2)
  Game 4: + 60/40 comrade healthcare reward.       (Phase 3)
  Game 5: + Gaussian fog on observations.          (Phase 4)
"""

import numpy as np
from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz


class KAZWrapper:
    """Parallel-env wrapper around KAZ with game-level modifiers."""

    # --- Reward weights per game level ---
    REWARD_CONFIGS = {
        1: {"weight_self": 1.0, "weight_team": 0.0},
        2: {"weight_self": 1.0, "weight_team": 0.0},
        3: {"weight_self": 1.0, "weight_team": 0.0},
        4: {"weight_self": 0.6, "weight_team": 0.4},
        5: {"weight_self": 0.6, "weight_team": 0.4},
    }

    def __init__(
        self,
        game_level: int = 1,
        max_cycles: int = 900,
        spawn_rate: int = 20,
        num_archers: int = 2,
        num_knights: int = 2,
        max_zombies: int = 10,
        max_arrows: int = 10,
        vector_state: bool = True,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        assert 1 <= game_level <= 5, f"game_level must be 1-5, got {game_level}"
        self.game_level = game_level
        self.vector_state = vector_state
        self._initial_seed = seed
        self._seeded = False

        # Feature flags (activated by game level)
        self.ammo_limit_enabled = game_level >= 2
        self.stamina_enabled = game_level >= 3
        self.team_reward_enabled = game_level >= 4
        self.fog_enabled = game_level >= 5

        # Reward weights
        cfg = self.REWARD_CONFIGS[game_level]
        self.weight_self = cfg["weight_self"]
        self.weight_team = cfg["weight_team"]

        # === Ammo system (Game 2+) ===
        self.max_ammo_per_archer = 15  # arrows per episode, tunable
        self.archer_ammo = {}         # runtime ammo counters
        self.dry_fire_penalty = -0.5

        # === Stamina system (Game 3+) ===
        self.stamina_cost_move = 0.01   # cost per movement action
        self.stamina_cost_attack = 0.05 # cost per sword swing
        self.agent_stamina = {}         # runtime stamina counters

        # === Fog system (Game 5) ===
        self.fog_sigma = 3.0  # Gaussian blur sigma

        # Create underlying PettingZoo parallel env
        self._env = kaz.parallel_env(
            spawn_rate=spawn_rate,
            num_archers=num_archers,
            num_knights=num_knights,
            max_zombies=max_zombies,
            max_arrows=max_arrows,
            killable_knights=True,
            killable_archers=True,
            pad_observation=True,
            line_death=False,
            max_cycles=max_cycles,
            vector_state=vector_state,
            render_mode=render_mode,
        )

    # ------------------------------------------------------------------
    # Delegated properties
    # ------------------------------------------------------------------
    @property
    def agents(self):
        return self._env.agents

    @property
    def possible_agents(self):
        return self._env.possible_agents

    @property
    def num_agents(self):
        return len(self._env.agents)

    def observation_space(self, agent):
        return self._env.observation_space(agent)

    def action_space(self, agent):
        return self._env.action_space(agent)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, seed=None):
        """Reset env and internal modifier state."""
        # Use initial seed only on the first reset for reproducibility;
        # subsequent resets let the internal RNG vary for diverse episodes.
        if seed is not None:
            reset_seed = seed
        elif not self._seeded and self._initial_seed is not None:
            reset_seed = self._initial_seed
            self._seeded = True
        else:
            reset_seed = None
        obs, infos = self._env.reset(seed=reset_seed)

        # Reset ammo counters
        self.archer_ammo = {
            a: self.max_ammo_per_archer
            for a in self._env.agents
            if a.startswith("archer")
        }
        # Reset stamina
        self.agent_stamina = {a: 1.0 for a in self._env.agents}

        return obs, infos

    def step(self, actions: dict):
        """
        Step the environment and apply game-level modifiers.

        Returns:
            obs, rewards, terminations, truncations, infos
            Each reward dict entry also populates infos[agent]["reward_info"]
            with decomposed components for TensorBoard logging.
        """
        # --- Pre-step modifiers (ammo interception, etc.) ---
        modified_actions = dict(actions)
        pre_penalties = {a: 0.0 for a in actions}

        if self.ammo_limit_enabled:
            pre_penalties = self._apply_ammo_limit(modified_actions, pre_penalties)

        # --- Env step ---
        obs, raw_rewards, terminations, truncations, infos = self._env.step(modified_actions)

        # --- Post-step reward shaping ---
        shaped_rewards = {}
        for agent in raw_rewards:
            reward_aggression = float(raw_rewards[agent]) + pre_penalties.get(agent, 0.0)
            reward_preservation = 0.0

            # Stamina cost (Game 3+)
            if self.stamina_enabled and agent in actions:
                stamina_cost = self._compute_stamina_cost(agent, actions[agent])
                reward_aggression -= stamina_cost

            # Team reward (Game 4+)
            if self.team_reward_enabled:
                # Average reward of teammates
                teammates = [a for a in raw_rewards if a != agent]
                if teammates:
                    reward_preservation = float(
                        np.mean([raw_rewards[t] for t in teammates])
                    )

            # Weighted combination
            total_reward = (
                self.weight_self * reward_aggression
                + self.weight_team * reward_preservation
            )
            shaped_rewards[agent] = total_reward

            # Attach decomposition for TensorBoard logging
            if agent not in infos:
                infos[agent] = {}
            infos[agent]["reward_info"] = {
                "raw_kill": float(raw_rewards[agent]),
                "aggression": reward_aggression,
                "preservation": reward_preservation,
                "total": total_reward,
            }

        # --- Fog (Game 5) applied to observations ---
        if self.fog_enabled:
            obs = self._apply_fog(obs)

        return obs, shaped_rewards, terminations, truncations, infos

    def close(self):
        self._env.close()

    def render(self):
        return self._env.render()

    # ------------------------------------------------------------------
    # Modifier helpers (stubs for Games 2-5, functional in later phases)
    # ------------------------------------------------------------------
    def _apply_ammo_limit(self, actions, penalties):
        """Game 2+: Intercept archer shoot actions when ammo is 0."""
        SHOOT_ACTION = 4  # PettingZoo KAZ: action 4 = fire arrow
        for agent in list(actions.keys()):
            if not agent.startswith("archer"):
                continue
            if agent not in self.archer_ammo:
                continue
            if actions[agent] == SHOOT_ACTION:
                if self.archer_ammo[agent] <= 0:
                    # Dry fire penalty
                    penalties[agent] += self.dry_fire_penalty
                    actions[agent] = 0  # Replace with no-op
                else:
                    self.archer_ammo[agent] -= 1
        return penalties

    def _compute_stamina_cost(self, agent, action):
        """Game 3+: Compute stamina cost for knight movement/attacks."""
        if not agent.startswith("knight"):
            return 0.0
        MOVE_ACTIONS = {0, 1, 2, 3}  # forward, back, rotate-left, rotate-right
        ATTACK_ACTION = 4              # weapon (slash)
        cost = 0.0
        if action in MOVE_ACTIONS:
            cost = self.stamina_cost_move
        elif action == ATTACK_ACTION:
            cost = self.stamina_cost_attack
        if agent in self.agent_stamina:
            self.agent_stamina[agent] = max(0.0, self.agent_stamina[agent] - cost)
        return cost

    def _apply_fog(self, obs):
        """Game 5: Apply Gaussian noise/blur to observations."""
        if self.vector_state:
            # For vector observations, add Gaussian noise
            fogged = {}
            for agent, ob in obs.items():
                noise = np.random.normal(0, self.fog_sigma * 0.1, size=ob.shape)
                fogged[agent] = (ob + noise).astype(ob.dtype)
            return fogged
        else:
            # For image observations, apply Gaussian blur (requires cv2)
            import cv2
            fogged = {}
            for agent, ob in obs.items():
                ksize = int(self.fog_sigma * 2) * 2 + 1
                fogged[agent] = cv2.GaussianBlur(ob, (ksize, ksize), self.fog_sigma)
            return fogged
