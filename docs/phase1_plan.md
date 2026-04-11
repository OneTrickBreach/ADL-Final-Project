# Phase 1: The Villain Baseline (Game 1)

## Objective
Train a Multi-Agent PPO (MAPPO) policy on the default PettingZoo Knights-Archers-Zombies (KAZ) environment with **no constraints** (infinite ammo, no stamina, perfect vision). The sole reward signal is **kills**. This produces the "Greedy Soldier" baseline against which all subsequent ablations (Games 2–5) are compared.

---

## Step-by-Step Development Plan

### Step 1 — Environment Scaffold
**Goal:** Verify PettingZoo KAZ runs, inspect its observation/action spaces, and create the base `KAZWrapper`.

| Sub-step | Detail |
|----------|--------|
| 1.1 | Install dependencies: `pettingzoo[butterfly]`, `supersuit`, `gymnasium`, `torch`, `tensorboard`, `stable-baselines3` (or custom PPO). Record versions in `requirements.txt`. |
| 1.2 | Write `src/wrappers/kaz_wrapper.py` — a thin wrapper class that accepts a `game_level` argument (1–5). For Game 1 the wrapper is essentially a **pass-through** (no modifications). Design the constructor so Games 2–5 can toggle ammo, stamina, fog via flags. |
| 1.3 | Write a small smoke-test script (`src/test_env.py`) that instantiates the wrapped env, runs 100 random-action steps, and prints observation shape, action space, and sample rewards. Run with `./.venv/bin/python src/test_env.py`. |
| 1.4 | Confirm CUDA availability inside the venv: `./.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`. |

**Artifacts:** `src/wrappers/__init__.py`, `src/wrappers/kaz_wrapper.py`, `src/test_env.py`, `requirements.txt`.

---

### Step 2 — MAPPO Network Definition
**Goal:** Define the shared-parameter actor-critic network with multi-objective reward heads.

| Sub-step | Detail |
|----------|--------|
| 2.1 | Create `src/models/__init__.py` and `src/models/mappo_net.py`. |
| 2.2 | Architecture: shared CNN/MLP backbone → two heads: **policy head** (actor, outputs action logits) and **value head** (critic, outputs state-value). |
| 2.3 | Add a `reward_decomposition` dict output from the value head so TensorBoard can log Aggression vs. Preservation components separately (even though Game 1 only uses Aggression = Kills). |
| 2.4 | Ensure the network is instantiated on `torch.device("cuda")`. |

**Artifacts:** `src/models/mappo_net.py`.

---

### Step 3 — Reward Logic (Game 1 Baseline)
**Goal:** Implement the reward scalarizer that will be extended in later phases.

| Sub-step | Detail |
|----------|--------|
| 3.1 | Inside `KAZWrapper.step()`, compute `reward_aggression` (raw kill reward from env) and `reward_preservation` (placeholder = 0 for Game 1). |
| 3.2 | Final reward: `R = weight_self * reward_aggression + weight_team * reward_preservation`. For Game 1, `weight_self = 1.0`, `weight_team = 0.0`. Store weights as wrapper attributes so Game 4 can switch to 0.6 / 0.4. |
| 3.3 | Return a `reward_info` dict alongside the scalar reward so the training loop can log both components to TensorBoard. |

**Artifacts:** Changes in `src/wrappers/kaz_wrapper.py`.

---

### Step 4 — Training Loop (`train.py`)
**Goal:** Implement the MAPPO training loop with full TensorBoard logging.

| Sub-step | Detail |
|----------|--------|
| 4.1 | Create `src/train.py` with argparse: `--game_level` (default 1), `--total_timesteps`, `--lr`, `--gamma`, `--gae_lambda`, `--clip_range`, `--n_epochs`, `--batch_size`, `--seed`. |
| 4.2 | Instantiate `KAZWrapper(game_level=1)`, vectorize with SuperSuit if needed. |
| 4.3 | Instantiate `MAPPONet` on CUDA, optimizer (Adam). |
| 4.4 | PPO loop: collect rollouts → compute GAE → update policy with clipped objective → update value function. |
| 4.5 | **TensorBoard logging (mandatory per rules.md #9):** Every `log_interval` steps, write: `reward/total`, `reward/aggression`, `reward/preservation`, `policy/entropy`, `policy/loss`, `value/loss`, `metrics/kill_density`, `metrics/survival_time`. Log dir: `results/tensorboard/game1/`. |
| 4.6 | Save checkpoints every `save_interval` steps to `models/game1/`. |

**Artifacts:** `src/train.py`, checkpoint dirs.

---

### Step 5 — Baseline Run & Metrics Collection
**Goal:** Execute the Game 1 training run and record baseline numbers.

| Sub-step | Detail |
|----------|--------|
| 5.1 | Run training: `./.venv/bin/python src/train.py --game_level 1 --total_timesteps 500000`. |
| 5.2 | Monitor via TensorBoard: `./.venv/bin/python -m tensorboard.main --logdir results/tensorboard/game1/ --port 6006`. |
| 5.3 | After training, extract and save baseline metrics to `results/game1_baseline_metrics.json`: **mean kill density**, **mean survival time**, **episode reward curve**. |
| 5.4 | Save final model to `models/game1/final.pt`. |

**Artifacts:** `results/tensorboard/game1/`, `results/game1_baseline_metrics.json`, `models/game1/final.pt`.

---

### Step 6 — Behavioral Validation
**Goal:** Confirm the "Greedy Soldier" behavior qualitatively.

| Sub-step | Detail |
|----------|--------|
| 6.1 | Write `src/evaluate.py` — loads a checkpoint, runs N evaluation episodes, records metrics. |
| 6.2 | Run evaluation and log: agents should show **overlapping paths, no coordination, aggressive charging**. |
| 6.3 | (Optional) Record a short video/gif of gameplay and save to `results/game1_demo/`. |

**Artifacts:** `src/evaluate.py`, `results/game1_demo/` (optional).

---

## Success Criteria for Phase 1
- [ ] KAZ env runs inside the wrapper with `game_level=1` on CUDA.
- [ ] MAPPO trains without errors for 500k+ timesteps.
- [ ] TensorBoard logs show reward decomposition (Aggression vs. Preservation).
- [ ] Baseline metrics (Kill Density, Survival Time) are saved to `results/`.
- [ ] Checkpoint saved to `models/game1/final.pt`.
- [ ] Agents exhibit "Greedy Soldier" behavior (no coordination, pure aggression).

---

## Execution Notes
- All commands run from project root (`~/ADLProject2`) per `rules.md`.
- All Python invocations use `./.venv/bin/python`.
- New pip packages → `./.venv/bin/pip install <pkg>` → update `requirements.txt`.
- Never delete `results/`, `models/`, or `ray_results/` without confirmation.
