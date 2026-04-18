"""KAZWrapperV3 — V3 wrapper implementing the 7-game progression from
docs/v3_psudoplan.txt and docs/v3_implementation_plan.md.

Supports game_level ∈ {"g0","g1a","g1b","g2","g3","g4","g5"}.

Key mechanics (see plan §4.1):
  - Ammo pool (global or individual), stamina pool for knights
  - Archer-immobile for G1a/G1b (pseudoplan §7a)
  - Roles (patrol / forward knight) for G3+
  - Target locks for G4+; pragmatic override for G5
  - Shared failure signal (+crossing detection)
  - Observation augmented with 5 extras per agent (role, in_end_zone,
    ammo_frac, stamina_frac, has_lock) → final obs = 140 dims.
"""

from __future__ import annotations

import math
import numpy as np
from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz

try:
    from pettingzoo.butterfly.knights_archers_zombies.src import constants as kaz_const
except Exception:
    kaz_const = None


# ── External action constants (verified in pre-flight) ──────────────
ACTION_FORWARD = 0
ACTION_BACKWARD = 1
ACTION_TURN_CCW = 2
ACTION_TURN_CW = 3
ACTION_ATTACK = 4      # fire arrow / sword swing
ACTION_NOOP = 5        # true no-op (internal 6)
ACTION_MOVEMENT_SET = {0, 1, 2, 3}

SCREEN_W = 1280
SCREEN_H = 720


GAME_CONFIGS = {
    "g0":  dict(ammo_on=False, ammo_mode=None,         stamina_on=False, archer_immobile=False, roles_on=False, lock_on=False, pragmatic=False),
    "g1a": dict(ammo_on=True,  ammo_mode="global",     stamina_on=True,  archer_immobile=True,  roles_on=False, lock_on=False, pragmatic=False),
    "g1b": dict(ammo_on=True,  ammo_mode="individual", stamina_on=True,  archer_immobile=True,  roles_on=False, lock_on=False, pragmatic=False),
    "g2":  dict(ammo_on=True,  ammo_mode=None,         stamina_on=True,  archer_immobile=False, roles_on=False, lock_on=False, pragmatic=False),
    "g3":  dict(ammo_on=True,  ammo_mode=None,         stamina_on=True,  archer_immobile=False, roles_on=True,  lock_on=False, pragmatic=False),
    "g4":  dict(ammo_on=True,  ammo_mode=None,         stamina_on=True,  archer_immobile=False, roles_on=True,  lock_on=True,  pragmatic=False),
    "g5":  dict(ammo_on=True,  ammo_mode=None,         stamina_on=True,  archer_immobile=False, roles_on=True,  lock_on=True,  pragmatic=True),
}

EXTRA_DIM = 5  # features appended per agent (see plan §4.1.7)


class KAZWrapperV3:
    """Parallel-env wrapper with V3 structured-teamwork mechanics."""

    def __init__(
        self,
        game_level: str,
        duration_seconds: int = 30,
        num_archers: int = 2,
        num_knights: int = 2,
        global_ammo_pool: int = 60,
        individual_ammo_pool: int = 30,
        knight_stamina_pool: int = 150,
        end_zone_fraction: float = 0.20,
        knight_lock_radius_fraction: float = 0.25,
        spawn_rate: int = 8,
        max_zombies: int = 20,
        ammo_mode_override: str | None = None,
        action_mask_mode: str = "soft",   # "soft" | "hard"
        invalid_action_penalty: float = 0.05,
        lock_bonus: float = 0.002,
        role_bonus: float = 0.001,
        failure_penalty: float = 1.0,
        vector_state: bool = True,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        assert game_level in GAME_CONFIGS, f"game_level must be one of {list(GAME_CONFIGS)}"
        self.game_level = game_level
        self.cfg = dict(GAME_CONFIGS[game_level])  # copy

        # Resolve ammo_mode (required for G2+ unless already fixed in config)
        if self.cfg["ammo_on"] and self.cfg["ammo_mode"] is None:
            if ammo_mode_override not in ("global", "individual"):
                raise ValueError(
                    f"game_level={game_level} requires ammo_mode_override "
                    f"∈ {{'global','individual'}}"
                )
            self.cfg["ammo_mode"] = ammo_mode_override

        # Config / hyperparams
        self.num_archers = num_archers
        self.num_knights = num_knights
        self.duration_seconds = duration_seconds
        self.global_ammo_pool = int(global_ammo_pool)
        self.individual_ammo_pool = int(individual_ammo_pool)
        self.knight_stamina_pool = int(knight_stamina_pool)
        self.end_zone_fraction = float(end_zone_fraction)
        self.knight_lock_radius_fraction = float(knight_lock_radius_fraction)
        self.action_mask_mode = action_mask_mode
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.lock_bonus = float(lock_bonus)
        self.role_bonus = float(role_bonus)
        self.failure_penalty = float(failure_penalty)

        # KAZ constants (with fallbacks per plan §4.1.8)
        self.ZOMBIE_Y_SPEED = int(getattr(kaz_const, "ZOMBIE_Y_SPEED", 5))
        self.KNIGHT_SPEED = int(getattr(kaz_const, "KNIGHT_SPEED", 25))
        self.SCREEN_W = int(getattr(kaz_const, "SCREEN_WIDTH", SCREEN_W))
        self.SCREEN_H = int(getattr(kaz_const, "SCREEN_HEIGHT", SCREEN_H))

        self.LOCK_RADIUS_PX = int(self.knight_lock_radius_fraction * self.SCREEN_W)
        self.END_ZONE_Y = int((1.0 - self.end_zone_fraction) * self.SCREEN_H)

        self.vector_state = vector_state
        self._initial_seed = seed
        self._seeded = False

        # Compute max_cycles from FPS (~15) × duration
        fps = int(getattr(kaz_const, "FPS", 15))
        self.max_cycles = int(duration_seconds * fps)

        # Build underlying env
        self._env = kaz.parallel_env(
            spawn_rate=spawn_rate,
            num_archers=num_archers,
            num_knights=num_knights,
            max_zombies=max_zombies,
            max_arrows=10,
            killable_knights=True,
            killable_archers=True,
            pad_observation=True,
            line_death=False,
            max_cycles=self.max_cycles,
            vector_state=vector_state,
            render_mode=render_mode,
        )

        # Runtime state (populated in reset)
        self.cycle = 0
        self.global_ammo = 0
        self.archer_ammo: dict[str, int] = {}
        self.knight_stamina: dict[str, int] = {}
        self.knight_role: dict[str, str] = {}
        self.lock_target: dict[str, int | None] = {}
        self.failure_count = 0
        self._crossed_ids: set[int] = set()
        self._prev_alive_count = 0
        self.attack_count: dict[str, int] = {}
        self.kill_count: dict[str, int] = {}
        self.last_reward_info: dict[str, dict] = {}
        self.last_failures_this_step = 0
        self.pragmatic_overrides = 0  # G5 diagnostic

    # ── Delegated properties ──────────────────────────────────
    @property
    def agents(self):
        return self._env.agents

    @property
    def possible_agents(self):
        return self._env.possible_agents

    @property
    def num_agents(self):
        return len(self._env.agents)

    @property
    def unwrapped_env(self):
        return self._env.unwrapped

    def observation_space(self, agent):
        return self._env.observation_space(agent)

    def action_space(self, agent):
        return self._env.action_space(agent)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    # ── Helpers ───────────────────────────────────────────────
    @staticmethod
    def _is_archer(name: str) -> bool:
        return name.startswith("archer")

    @staticmethod
    def _is_knight(name: str) -> bool:
        return name.startswith("knight")

    def _agent_position(self, agent_name: str):
        """Return (x, y) pixel position of a live player, or None."""
        raw = self._env.unwrapped
        if not hasattr(raw, "agent_name_mapping"):
            return None
        idx = raw.agent_name_mapping.get(agent_name)
        if idx is None:
            return None
        try:
            p = raw.agent_list[idx]
        except Exception:
            return None
        if not getattr(p, "alive", False):
            return None
        return (float(p.rect.centerx), float(p.rect.centery))

    def _zombies(self):
        raw = self._env.unwrapped
        out = []
        zl = getattr(raw, "zombie_list", None)
        if zl is None:
            return out
        for z in zl:
            try:
                out.append((id(z), float(z.rect.centerx), float(z.rect.centery), z))
            except Exception:
                continue
        return out

    # ── Lock acquisition (§4.1.8) ─────────────────────────────
    def _acquire_locks(self, zombies_snapshot):
        """Refresh self.lock_target for all live agents under G4+ rules."""
        if not self.cfg["lock_on"]:
            return

        zombie_map = {zid: (zx, zy) for zid, zx, zy, _ in zombies_snapshot}
        # Invalidate dead/missing locks
        for a in list(self.lock_target.keys()):
            tid = self.lock_target.get(a)
            if tid is not None and tid not in zombie_map:
                self.lock_target[a] = None

        # Current knight locks
        knight_locks = {
            a: self.lock_target[a]
            for a in self.lock_target
            if self._is_knight(a) and self.lock_target.get(a) is not None
        }
        knight_locked_ids = set(tid for tid in knight_locks.values() if tid is not None)

        for agent in list(self.lock_target.keys()):
            if self.lock_target.get(agent) is not None:
                continue
            pos = self._agent_position(agent)
            if pos is None:
                continue
            ax, ay = pos

            if self._is_archer(agent):
                cands = [(zid, zx, zy) for zid, zx, zy, _ in zombies_snapshot
                         if zid not in knight_locked_ids]
                pragmatic_ids = set()
                if self.cfg["pragmatic"]:
                    for kagent, ktid in knight_locks.items():
                        if ktid is None or ktid not in zombie_map:
                            continue
                        zx, zy = zombie_map[ktid]
                        kpos = self._agent_position(kagent)
                        if kpos is None:
                            continue
                        cross_time = (self.SCREEN_H - zy) / max(self.ZOMBIE_Y_SPEED, 1)
                        inter_time = math.hypot(kpos[0] - zx, kpos[1] - zy) / max(self.KNIGHT_SPEED, 1)
                        if cross_time < inter_time:
                            cands.append((ktid, zx, zy))
                            pragmatic_ids.add(ktid)
                if cands:
                    chosen = min(cands, key=lambda t: math.hypot(t[1] - ax, t[2] - ay))
                    self.lock_target[agent] = chosen[0]
                    if chosen[0] in pragmatic_ids:
                        self.pragmatic_overrides += 1
                else:
                    self.lock_target[agent] = None

            elif self._is_knight(agent):
                is_patrol = (self.knight_role.get(agent) == "patrol")
                in_range = []
                for zid, zx, zy, _ in zombies_snapshot:
                    if math.hypot(zx - ax, zy - ay) > self.LOCK_RADIUS_PX:
                        continue
                    if is_patrol and zy < self.END_ZONE_Y:
                        continue
                    in_range.append((zid, zx, zy))
                if in_range:
                    zid, zx, zy = min(in_range, key=lambda t: math.hypot(t[1] - ax, t[2] - ay))
                    self.lock_target[agent] = zid
                else:
                    self.lock_target[agent] = None

    # ── Reset ─────────────────────────────────────────────────
    def reset(self, seed=None):
        if seed is not None:
            rs = seed
        elif not self._seeded and self._initial_seed is not None:
            rs = self._initial_seed
            self._seeded = True
        else:
            rs = None
        obs, infos = self._env.reset(seed=rs)

        self.cycle = 0
        self.failure_count = 0
        self.pragmatic_overrides = 0
        self.last_failures_this_step = 0
        self._crossed_ids = set()
        raw = self._env.unwrapped
        self._prev_alive_count = (
            len(getattr(raw, "archer_list", []))
            + len(getattr(raw, "knight_list", []))
        )

        agents = list(self._env.agents)
        archers = [a for a in agents if self._is_archer(a)]
        knights = [a for a in agents if self._is_knight(a)]

        self.global_ammo = self.global_ammo_pool
        self.archer_ammo = {a: self.individual_ammo_pool for a in archers}
        self.knight_stamina = {k: self.knight_stamina_pool for k in knights}

        self.knight_role = {}
        if self.cfg["roles_on"] and len(knights) >= 2:
            # Deterministic assignment (matches plan)
            self.knight_role = {knights[0]: "forward", knights[1]: "patrol"}
        else:
            # Default roles (affect obs only if roles_on True)
            for k in knights:
                self.knight_role[k] = "forward"

        self.lock_target = {a: None for a in agents}
        self.attack_count = {a: 0 for a in agents}
        self.kill_count = {a: 0 for a in agents}
        self.last_reward_info = {a: {} for a in agents}

        obs_aug = self._augment_obs(obs)
        return obs_aug, infos

    # ── Ammo / stamina bookkeeping ────────────────────────────
    def _archer_can_fire(self, a: str) -> bool:
        if not self.cfg["ammo_on"]:
            return True
        if self.cfg["ammo_mode"] == "global":
            return self.global_ammo > 0
        return self.archer_ammo.get(a, 0) > 0

    def _spend_archer_ammo(self, a: str):
        if not self.cfg["ammo_on"]:
            return
        if self.cfg["ammo_mode"] == "global":
            self.global_ammo = max(0, self.global_ammo - 1)
        else:
            self.archer_ammo[a] = max(0, self.archer_ammo.get(a, 0) - 1)

    def _knight_can_act(self, k: str) -> bool:
        if not self.cfg["stamina_on"]:
            return True
        return self.knight_stamina.get(k, 0) > 0

    def _spend_knight_stamina(self, k: str):
        if not self.cfg["stamina_on"]:
            return
        self.knight_stamina[k] = max(0, self.knight_stamina.get(k, 0) - 1)

    # ── Step ──────────────────────────────────────────────────
    def step(self, actions: dict):
        cfg = self.cfg
        raw = self._env.unwrapped

        modified = dict(actions)
        invalid_flags = {a: False for a in actions}   # soft-penalty flags
        lock_respected = {a: False for a in actions}  # for G4+ bonus

        # Snapshot zombie positions BEFORE step (for cross detection + lock acquisition)
        zombies_snapshot = self._zombies()
        if cfg["lock_on"]:
            # Update locks BEFORE action masking so that masking can use them
            self._acquire_locks(zombies_snapshot)

        # Apply constraints/masks
        for a, act in list(modified.items()):
            orig = act
            is_archer = self._is_archer(a)
            is_knight = self._is_knight(a)

            if orig == ACTION_ATTACK:
                self.attack_count[a] = self.attack_count.get(a, 0) + 1

            # (b) Archer-immobile
            if is_archer and cfg["archer_immobile"] and orig in ACTION_MOVEMENT_SET:
                invalid_flags[a] = True
                modified[a] = ACTION_NOOP
                continue

            # (c) Patrol role constraint (G4+): direction-aware hard mask that
            # prevents a patrol knight from crossing ABOVE the end-zone line.
            # Because KAZ movement direction depends on the agent's current
            # orientation vector `direction=(dx,dy)`, we project the step onto
            # y and only mask when the resulting y would leave the end zone.
            if (is_knight and cfg["roles_on"] and cfg["lock_on"]
                    and self.knight_role.get(a) == "patrol"
                    and orig in (ACTION_FORWARD, ACTION_BACKWARD)):
                try:
                    idx = raw.agent_name_mapping.get(a)
                    p = raw.agent_list[idx] if idx is not None else None
                    if p is not None and getattr(p, "alive", False):
                        dy = float(p.direction.y)
                        sign = -1.0 if orig == ACTION_FORWARD else 1.0
                        # Predicted y change this step (forward uses +dir, backward uses -dir)
                        # Player.update uses rect.y -= sin(angle+90)*speed for forward,
                        # rect.y += sin(angle+90)*speed for backward.
                        # direction.y = -sin(angle+90), so predicted dy = sign * (-dy) * speed.
                        # Sign of predicted delta-y = sign * (-dy); negative means moving up.
                        predicted_dy = sign * (-dy) * float(p.speed)
                        ay_now = float(p.rect.centery)
                        if ay_now >= self.END_ZONE_Y and (ay_now + predicted_dy) < self.END_ZONE_Y:
                            invalid_flags[a] = True
                            modified[a] = ACTION_NOOP
                            continue
                except Exception:
                    pass

            # (d) Ammo check
            if is_archer and modified[a] == ACTION_ATTACK:
                if not self._archer_can_fire(a):
                    invalid_flags[a] = True
                    modified[a] = ACTION_NOOP
                else:
                    self._spend_archer_ammo(a)

            # (e) Stamina: any non-noop action for knights
            if is_knight:
                if modified[a] != ACTION_NOOP:
                    if not self._knight_can_act(a):
                        invalid_flags[a] = True
                        modified[a] = ACTION_NOOP
                    else:
                        self._spend_knight_stamina(a)

            # (f) Lock mask (G4+)
            if cfg["lock_on"] and modified[a] == ACTION_ATTACK:
                target = self.lock_target.get(a)
                if target is None:
                    invalid_flags[a] = True
                    modified[a] = ACTION_NOOP
                else:
                    # Knight: target must be within lock radius
                    if is_knight:
                        pos = self._agent_position(a)
                        zmap = {zid: (zx, zy) for zid, zx, zy, _ in zombies_snapshot}
                        tpos = zmap.get(target)
                        if pos is None or tpos is None:
                            invalid_flags[a] = True
                            modified[a] = ACTION_NOOP
                        else:
                            if math.hypot(tpos[0] - pos[0], tpos[1] - pos[1]) > self.LOCK_RADIUS_PX:
                                invalid_flags[a] = True
                                modified[a] = ACTION_NOOP
                            else:
                                lock_respected[a] = True
                    else:
                        lock_respected[a] = True  # archer firing with a valid lock

        # Env step
        obs, raw_rewards, terms, truncs, infos = self._env.step(modified)

        # Kill attribution
        for a, r in raw_rewards.items():
            if r > 0.5:
                self.kill_count[a] = self.kill_count.get(a, 0) + 1

        # --- V3.1 failure detector: line-cross + agent-breach -----------
        raw = self._env.unwrapped
        CROSS_Y = self.SCREEN_H - self.ZOMBIE_Y_SPEED

        failures_this_step = 0

        # (a) zombies past the end line (still in list; KAZ does not remove crossers)
        for z in list(getattr(raw, "zombie_list", [])):
            try:
                zid = id(z)
                if z.rect.centery >= CROSS_Y and zid not in self._crossed_ids:
                    self._crossed_ids.add(zid)
                    failures_this_step += 1
            except Exception:
                continue

        # (b) agents killed by zombie contact this step (archer/knight lists shrink)
        cur_alive = (
            len(getattr(raw, "archer_list", []))
            + len(getattr(raw, "knight_list", []))
        )
        if cur_alive < self._prev_alive_count:
            failures_this_step += (self._prev_alive_count - cur_alive)
        self._prev_alive_count = cur_alive

        self.failure_count += failures_this_step
        self.last_failures_this_step = failures_this_step

        # --- V3.1 tier-2: episode-continuation override -----------------
        # Allow episode to continue past individual line crosses so the
        # policy can learn from multiple failures per rollout. Only
        # override when agents are still alive and we have not hit the
        # natural truncation.
        if (
            not getattr(raw, "run", True)
            and cur_alive > 0
            and self.cycle + 1 < self.max_cycles
        ):
            raw.run = True
            terms = {a: False for a in terms}
            truncs = {a: False for a in truncs}
            # Prevent the just-counted crossers from re-triggering
            # zombie_endscreen on the very next step.
            for z in list(getattr(raw, "zombie_list", [])):
                try:
                    if id(z) in self._crossed_ids and z.rect.centery >= CROSS_Y:
                        raw.zombie_list.remove(z)
                except Exception:
                    continue

        # Refresh locks after step (targets may have died)
        new_zombies = self._zombies()
        if cfg["lock_on"]:
            self._acquire_locks(new_zombies)

        # Compute shaped rewards
        shaped = {}
        living_agents = list(raw_rewards.keys())
        n_alive = max(len(living_agents), 1)
        for a in raw_rewards:
            raw_k = float(raw_rewards[a])
            role_b = 0.0
            lock_b = 0.0
            pen = 0.0
            # Preemptive cap (plan §9 Risk #3): clamp reward term at 3/step
            # to keep training stable; raw failure_count is unclamped.
            failure_share = -self.failure_penalty * min(failures_this_step, 3)

            if cfg["roles_on"] and self._is_knight(a):
                pos = self._agent_position(a)
                if pos is not None:
                    in_ez = 1.0 if pos[1] >= self.END_ZONE_Y else 0.0
                else:
                    in_ez = 0.0
                role = self.knight_role.get(a, "forward")
                if role == "patrol":
                    role_b = self.role_bonus * in_ez
                else:
                    role_b = self.role_bonus * (1.0 - in_ez)

            if cfg["lock_on"] and lock_respected.get(a, False):
                lock_b = self.lock_bonus

            if self.action_mask_mode == "soft" and invalid_flags.get(a, False):
                pen = -self.invalid_action_penalty

            total = raw_k + role_b + lock_b + failure_share + pen
            shaped[a] = total

            if a not in infos:
                infos[a] = {}
            infos[a]["reward_info"] = {
                "raw_kill": raw_k,
                "role_bonus": role_b,
                "lock_bonus": lock_b,
                "failure_shared": failure_share,
                "invalid_penalty": pen,
                "total": total,
                # legacy keys so existing train.py-style loggers still work
                "aggression": raw_k,
                "preservation": role_b + lock_b,
                "death_penalty": 0.0,
            }
            self.last_reward_info[a] = infos[a]["reward_info"]

        self.cycle += 1
        obs_aug = self._augment_obs(obs)
        return obs_aug, shaped, terms, truncs, infos

    # ── Observation augmentation (§4.1.7) ─────────────────────
    def _augment_obs(self, obs_dict):
        """Append 5 extras per agent → 135+5 = 140 dims (flattened view).

        We preserve the (27, 5) base by appending a (1, 5) row at the end
        so the flattened obs stays 140. The model reshapes the first 135
        values to (27, 5) and treats the trailing 5 as the extras vector.
        """
        cfg = self.cfg
        out = {}
        for a, ob in obs_dict.items():
            # Defaults
            role_id = 0.0
            if cfg["roles_on"] and self._is_knight(a):
                role_id = 1.0 if self.knight_role.get(a) == "patrol" else 0.0

            pos = self._agent_position(a)
            if pos is not None:
                in_ez = 1.0 if pos[1] >= self.END_ZONE_Y else 0.0
            else:
                in_ez = 0.0

            if self._is_archer(a) and cfg["ammo_on"]:
                if cfg["ammo_mode"] == "global":
                    ammo_frac = self.global_ammo / max(self.global_ammo_pool, 1)
                else:
                    ammo_frac = self.archer_ammo.get(a, 0) / max(self.individual_ammo_pool, 1)
            elif self._is_archer(a):
                ammo_frac = 1.0
            else:
                ammo_frac = 0.0

            if self._is_knight(a) and cfg["stamina_on"]:
                stamina_frac = self.knight_stamina.get(a, 0) / max(self.knight_stamina_pool, 1)
            elif self._is_knight(a):
                stamina_frac = 1.0
            else:
                stamina_frac = 0.0

            has_lock = 1.0 if (cfg["lock_on"] and self.lock_target.get(a) is not None) else 0.0

            extras = np.array([role_id, in_ez, ammo_frac, stamina_frac, has_lock],
                              dtype=np.float32)

            if isinstance(ob, np.ndarray) and ob.ndim == 2:
                # (27, 5) → append row of 5 → (28, 5)
                extras_row = extras.reshape(1, -1).astype(ob.dtype)
                out[a] = np.concatenate([ob, extras_row], axis=0)
            else:
                base = np.asarray(ob, dtype=np.float32).flatten()
                out[a] = np.concatenate([base, extras], axis=0)
        return out

    # ── Diagnostics for eval ──────────────────────────────────
    def get_episode_stats(self):
        archers = [a for a in self.possible_agents if self._is_archer(a)]
        knights = [a for a in self.possible_agents if self._is_knight(a)]
        ammo_used_per_archer = []
        if self.cfg["ammo_on"]:
            if self.cfg["ammo_mode"] == "individual":
                for a in archers:
                    used = self.individual_ammo_pool - self.archer_ammo.get(a, 0)
                    ammo_used_per_archer.append(used / max(self.individual_ammo_pool, 1))
                ammo_pct_expended = float(np.mean(ammo_used_per_archer)) if ammo_used_per_archer else 0.0
            else:
                ammo_pct_expended = (self.global_ammo_pool - self.global_ammo) / max(self.global_ammo_pool, 1)
                ammo_used_per_archer = [None] * len(archers)
        else:
            ammo_pct_expended = 0.0
            ammo_used_per_archer = [None] * len(archers)

        stamina_per_knight = []
        for k in knights:
            used = self.knight_stamina_pool - self.knight_stamina.get(k, self.knight_stamina_pool)
            stamina_per_knight.append(used / max(self.knight_stamina_pool, 1))
        stamina_pct_expended = float(np.mean(stamina_per_knight)) if stamina_per_knight else 0.0

        kills_archer = sum(self.kill_count.get(a, 0) for a in archers)
        kills_knight = sum(self.kill_count.get(k, 0) for k in knights)
        attacks_archer = sum(self.attack_count.get(a, 0) for a in archers)
        attacks_knight = sum(self.attack_count.get(k, 0) for k in knights)
        total_kills = kills_archer + kills_knight
        score = total_kills - self.failure_count

        return {
            "score": int(score),
            "kills_total": int(total_kills),
            "kills_archer": int(kills_archer),
            "kills_knight": int(kills_knight),
            "attacks_archer": int(attacks_archer),
            "attacks_knight": int(attacks_knight),
            "failures": int(self.failure_count),
            "episode_length": int(self.cycle),
            "ammo_pct_expended": float(ammo_pct_expended),
            "ammo_pct_expended_per_archer": ammo_used_per_archer,
            "stamina_pct_expended": float(stamina_pct_expended),
            "stamina_pct_expended_per_knight": stamina_per_knight,
            "pragmatic_overrides": int(self.pragmatic_overrides),
            "ammo_mode": self.cfg.get("ammo_mode"),
        }
