"""
Smoke test — verifies all core components work before real development.
Tests: imports, KAZ env rollout, torch+CUDA, network forward pass, TensorBoard.
Run with: .venv/bin/python test/smoke_test.py
"""

import sys

PASS = "[PASS]"
FAIL = "[FAIL]"


def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


# ── 1. Imports ────────────────────────────────────────
section("1. Import Check")

try:
    import torch
    print(f"{PASS} torch {torch.__version__}")
except Exception as e:
    print(f"{FAIL} torch: {e}"); sys.exit(1)

try:
    import numpy as np
    print(f"{PASS} numpy {np.__version__}")
except Exception as e:
    print(f"{FAIL} numpy: {e}")

try:
    from pettingzoo.butterfly.knights_archers_zombies import knights_archers_zombies as kaz
    print(f"{PASS} pettingzoo KAZ")
except Exception as e:
    print(f"{FAIL} pettingzoo: {e}"); sys.exit(1)

try:
    import gymnasium
    print(f"{PASS} gymnasium {gymnasium.__version__}")
except Exception as e:
    print(f"{FAIL} gymnasium: {e}")

try:
    import supersuit
    print(f"{PASS} supersuit")
except Exception as e:
    print(f"{FAIL} supersuit: {e}")

try:
    import cv2
    print(f"{PASS} opencv-python {cv2.__version__}")
except Exception as e:
    print(f"{FAIL} opencv: {e}")

try:
    import matplotlib
    print(f"{PASS} matplotlib {matplotlib.__version__}")
except Exception as e:
    print(f"{FAIL} matplotlib: {e}")

try:
    from torch.utils.tensorboard import SummaryWriter
    print(f"{PASS} tensorboard SummaryWriter")
except Exception as e:
    print(f"{FAIL} tensorboard: {e}")

try:
    import ray
    print(f"{PASS} ray {ray.__version__}")
except Exception as e:
    print(f"{FAIL} ray: {e}")


# ── 2. CUDA ───────────────────────────────────────────
section("2. CUDA / GPU Check")

cuda_ok = torch.cuda.is_available()
print(f"{'PASS' if cuda_ok else 'WARN'} CUDA available: {cuda_ok}")
if cuda_ok:
    print(f"{PASS} GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("  [WARN] Falling back to CPU for this test.")
    device = torch.device("cpu")


# ── 3. KAZ Environment Rollout ────────────────────────
section("3. KAZ Environment — Random Rollout (50 steps)")

obs_shapes, act_spaces = {}, {}
try:
    env = kaz.env(render_mode=None)
    env.reset(seed=42)

    obs_shapes, act_spaces = {}, {}
    for agent in env.agents:
        obs_shapes[agent] = env.observation_space(agent).shape
        act_spaces[agent] = env.action_space(agent).n
    print(f"{PASS} Agents: {list(obs_shapes.keys())}")
    print(f"      Obs shapes : {obs_shapes}")
    print(f"      Act spaces : {act_spaces}")

    steps = 0
    for agent in env.agent_iter(max_iter=50):
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            action = None
        else:
            action = env.action_space(agent).sample()
        env.step(action)
        steps += 1

    env.close()
    print(f"{PASS} Rollout completed ({steps} steps, no crash)")
except Exception as e:
    print(f"{FAIL} KAZ rollout: {e}")
    import traceback; traceback.print_exc()


# ── 4. Minimal Network Forward Pass ──────────────────
section("4. Minimal Policy Network — Forward Pass")

try:
    import torch.nn as nn

    first_agent = list(obs_shapes.keys())[0]
    obs_dim = int(np.prod(obs_shapes[first_agent]))
    act_dim = act_spaces[first_agent]

    policy = nn.Sequential(
        nn.Linear(obs_dim, 128),
        nn.ReLU(),
        nn.Linear(128, act_dim)
    ).to(device)

    dummy_obs = torch.zeros(1, obs_dim).to(device)
    logits = policy(dummy_obs)
    print(f"{PASS} Forward pass OK | obs_dim={obs_dim} → logits shape {tuple(logits.shape)}")
except Exception as e:
    print(f"{FAIL} Network forward pass: {e}")
    import traceback; traceback.print_exc()


# ── 5. TensorBoard Writer ─────────────────────────────
section("5. TensorBoard SummaryWriter")

try:
    import os
    log_dir = "results/tensorboard/smoke_test"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_scalar("test/reward", 1.0, global_step=0)
    writer.add_scalar("test/loss", 0.5, global_step=0)
    writer.close()
    print(f"{PASS} SummaryWriter wrote to {log_dir}")
except Exception as e:
    print(f"{FAIL} TensorBoard writer: {e}")


# ── Summary ───────────────────────────────────────────
section("Smoke Test Complete")
print("If all PASS above, the environment is ready for development.\n")
