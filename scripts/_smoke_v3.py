"""Final smoke test: instantiate KAZWrapperV3 for each game level,
run 5 random-action steps, verify no crashes and obs shape == 140."""
import sys, traceback
sys.path.insert(0, ".")
import numpy as np
from src.wrappers.kaz_wrapper_v3 import KAZWrapperV3
from src.policies.heuristic import heuristic_actions_all

GAMES = ["g0", "g1a", "g1b", "g2", "g3", "g4", "g5"]

for g in GAMES:
    try:
        env = KAZWrapperV3(
            game_level=g,
            ammo_mode_override=("individual" if g in ("g2","g3","g4","g5") else None),
            seed=42,
        )
        obs, _ = env.reset()
        first = list(obs.values())[0]
        flat = first.flatten()
        assert flat.shape[0] == 140, f"expected 140, got {flat.shape}"
        # 5 steps random
        rng = np.random.default_rng(0)
        for i in range(5):
            if not env.agents: break
            acts = {a: int(rng.integers(0, 6)) for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(acts)
        # Also verify heuristic
        if g in ("g0","g1a","g1b"):
            env2 = KAZWrapperV3(game_level=g, seed=42)
            env2.reset()
            for i in range(5):
                if not env2.agents: break
                acts = heuristic_actions_all(env2)
                env2.step(acts)
            env2.close()
        stats = env.get_episode_stats()
        env.close()
        print(f"OK {g}: obs_dim=140 steps_ok stats_keys={len(stats)} "
              f"ammo_mode={stats.get('ammo_mode')}")
    except Exception as e:
        print(f"FAIL {g}: {e}")
        traceback.print_exc()
        sys.exit(1)

print("\nAll 7 games smoke-tested OK.")
