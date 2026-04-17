"""V3 pre-flight smoke tests (Section 9). Runs in <60s."""
import sys, traceback
sys.path.insert(0, ".")
import numpy as np
from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz

results = {}

# 2. FPS / max_cycles
try:
    env = kaz.parallel_env(vector_state=True, pad_observation=True,
                           line_death=False, killable_archers=True,
                           killable_knights=True, max_cycles=450,
                           num_archers=2, num_knights=2)
    env.reset(seed=0)
    md = env.metadata
    results["fps_metadata"] = md.get("render_fps")
    raw = env.unwrapped
    # 3. zombie_list
    print("raw dir filtered:", [x for x in dir(raw) if "zomb" in x.lower() or "screen" in x.lower() or "const" in x.lower()][:20])
    results["has_zombie_list"] = hasattr(raw, "zombie_list")
    results["zombie_list_type"] = type(getattr(raw, "zombie_list", None)).__name__
    # 4. screen dims — try various
    if hasattr(raw, "screen"):
        try:
            results["screen_size"] = raw.screen.get_size()
        except Exception as e:
            results["screen_size_err"] = str(e)
    # Check width/height attrs
    for attr in ["WIDTH", "HEIGHT", "SCREEN_WIDTH", "SCREEN_HEIGHT"]:
        if hasattr(raw, attr):
            results[f"raw.{attr}"] = getattr(raw, attr)
    # Observation shape (5. obs)
    obs, _ = env.reset(seed=0)
    first = list(obs.values())[0]
    results["obs_shape"] = first.shape
    results["num_agents"] = len(env.agents)
    results["agents"] = list(env.agents)
    results["act_space_n"] = env.action_space(env.agents[0]).n
    # 1. Action mapping — just confirm action 4 is valid (actual motion check deferred)
    # Step with no-op and record
    step_acts = {a: 0 for a in env.agents}
    env.step(step_acts)
    results["step0_ok"] = True
except Exception as e:
    traceback.print_exc()
    results["err"] = str(e)
finally:
    try: env.close()
    except: pass

# 7. Constants
try:
    from pettingzoo.butterfly.knights_archers_zombies import constants as kconst
    results["consts_module"] = "constants"
except Exception:
    try:
        from pettingzoo.butterfly.knights_archers_zombies import const as kconst
        results["consts_module"] = "const"
    except Exception as e:
        kconst = None
        results["consts_err"] = str(e)
if kconst is not None:
    for name in dir(kconst):
        if name.isupper():
            val = getattr(kconst, name)
            if isinstance(val, (int, float, str, tuple, list)):
                results[f"const.{name}"] = val

for k, v in results.items():
    print(f"{k} = {v}")
