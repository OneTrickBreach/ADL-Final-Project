"""Smoke test: instantiate KAZWrapper, run 100 steps with random actions."""

import sys
import numpy as np

sys.path.insert(0, ".")
from src.wrappers.kaz_wrapper import KAZWrapper


def main():
    env = KAZWrapper(game_level=1, max_cycles=200, vector_state=True)
    obs, infos = env.reset(seed=42)

    print("=" * 60)
    print("KAZWrapper Smoke Test  (game_level=1)")
    print("=" * 60)
    print(f"Agents: {env.agents}")
    for a in env.agents:
        print(f"  {a}: obs_shape={obs[a].shape}, action_space={env.action_space(a)}")
    print()

    total_rewards = {a: 0.0 for a in env.agents}
    steps = 0

    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for a in rewards:
            total_rewards[a] += rewards[a]
        steps += 1
        if steps <= 3:
            # Show reward_info for first few steps
            sample_agent = list(infos.keys())[0]
            ri = infos[sample_agent].get("reward_info", {})
            print(f"Step {steps}: reward_info({sample_agent}) = {ri}")

    print()
    print(f"Episode finished in {steps} steps")
    print(f"Total rewards: {total_rewards}")
    env.close()
    print("Smoke test PASSED")


if __name__ == "__main__":
    main()
