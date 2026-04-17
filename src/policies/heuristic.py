"""Scripted heuristic policy for G0/G1a/G1b (no learning).

Uses the KAZWrapperV3's underlying env to query true agent/zombie positions
rather than parsing the (27, 5) observation, which keeps the policy robust
to KAZ version changes.

API:
    action = heuristic_action(agent_name, wrapper)

Behaviour:
    - Archer: aim at nearest zombie; fire if ammo > 0 and roughly aligned.
              Never move if cfg["archer_immobile"] is True; otherwise rotate
              toward the zombie before firing.
    - Knight: walk toward nearest zombie; attack when within ~60 px.
"""

from __future__ import annotations

import math
import numpy as np

from src.wrappers.kaz_wrapper_v3 import (
    ACTION_FORWARD, ACTION_TURN_CCW, ACTION_TURN_CW,
    ACTION_ATTACK, ACTION_NOOP,
)

_ATTACK_RADIUS = 60
_AIM_TOLERANCE_RAD = 0.35  # ~20° — archer shoots when target is roughly forward


def _nearest_zombie(pos, zombies):
    if not zombies:
        return None
    ax, ay = pos
    return min(zombies, key=lambda z: math.hypot(z[1] - ax, z[2] - ay))


def _direction_vec(agent):
    """Return unit vector of agent's facing direction (pygame Vector2)."""
    try:
        d = agent.direction
        return float(d.x), float(d.y)
    except Exception:
        return 0.0, -1.0


def heuristic_action(agent_name: str, wrapper) -> int:
    raw = wrapper.unwrapped_env
    idx = raw.agent_name_mapping.get(agent_name)
    if idx is None:
        return ACTION_NOOP
    try:
        agent = raw.agent_list[idx]
    except Exception:
        return ACTION_NOOP
    if not getattr(agent, "alive", False):
        return ACTION_NOOP

    pos = (float(agent.rect.centerx), float(agent.rect.centery))
    zombies = wrapper._zombies()
    if not zombies:
        return ACTION_NOOP

    target = _nearest_zombie(pos, zombies)
    if target is None:
        return ACTION_NOOP

    tid, tx, ty, _ = target
    dx, dy = tx - pos[0], ty - pos[1]
    dist = math.hypot(dx, dy)

    is_archer = wrapper._is_archer(agent_name)
    is_knight = wrapper._is_knight(agent_name)

    if is_archer:
        # Out of ammo → noop
        if wrapper.cfg["ammo_on"] and not wrapper._archer_can_fire(agent_name):
            return ACTION_NOOP

        ux, uy = _direction_vec(agent)
        tlen = max(math.hypot(dx, dy), 1e-6)
        cos_theta = (ux * dx + uy * dy) / tlen
        # If approximately facing target, attack
        if cos_theta > math.cos(_AIM_TOLERANCE_RAD):
            return ACTION_ATTACK

        # Otherwise rotate toward target (cross-product sign picks direction)
        cross = ux * dy - uy * dx
        # pygame y grows downward; KAZ rotates "+angle" as CCW in screen terms.
        # Empirically: turn CCW if cross < 0 else CW (direction is relative but
        # the agent receives immediate feedback, so this self-corrects quickly).
        if wrapper.cfg["archer_immobile"]:
            return ACTION_TURN_CCW if cross < 0 else ACTION_TURN_CW
        return ACTION_TURN_CCW if cross < 0 else ACTION_TURN_CW

    if is_knight:
        if wrapper.cfg["stamina_on"] and not wrapper._knight_can_act(agent_name):
            return ACTION_NOOP

        if dist <= _ATTACK_RADIUS:
            return ACTION_ATTACK

        ux, uy = _direction_vec(agent)
        tlen = max(math.hypot(dx, dy), 1e-6)
        cos_theta = (ux * dx + uy * dy) / tlen
        if cos_theta > math.cos(0.5):
            return ACTION_FORWARD
        cross = ux * dy - uy * dx
        return ACTION_TURN_CCW if cross < 0 else ACTION_TURN_CW

    return ACTION_NOOP


def heuristic_actions_all(wrapper) -> dict:
    """Convenience: returns dict of actions for all currently-living agents."""
    return {a: heuristic_action(a, wrapper) for a in wrapper.agents}
