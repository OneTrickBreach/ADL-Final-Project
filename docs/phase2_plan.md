# Phase 2: Resource Scarcity — Game 2 (Ammo Restriction) & Game 3 (Stamina Decay)

**Branch:** `phase2-resource-scarcity`
**Goal:** Train agents under resource constraints and observe the emergence of *Trigger Discipline* (G2) and *Economic Positioning* (G3).

---

## Prerequisites

- Phase 1 complete — `models/game1/final.pt` exists with baseline metrics.
- `KAZWrapper` already implements ammo limit (Game 2+) and stamina decay (Game 3+).
- Virtual environment at `.venv/` with all dependencies from `requirements.txt`.

### Phase 1 Baseline Reference

| Metric | Value |
|--------|-------|
| Mean return | 3.93 ± 3.91 |
| Kill density | 0.0102 |
| Mean ep length | 386 steps |
| Total steps trained | 501,838 |
| Checkpoint | `models/game1/final.pt` |

---

## Wrapper Mechanics Already in Place

### Game 2 — Ammo Restriction (`src/wrappers/kaz_wrapper.py`)

| Parameter | Value | Location |
|-----------|-------|----------|
| `ammo_limit_enabled` | `True` when `game_level >= 2` | `KAZWrapper.__init__` |
| `max_ammo_per_archer` | 15 arrows per episode | `KAZWrapper.__init__` |
| `dry_fire_penalty` | −0.5 per dry fire | `KAZWrapper.__init__` |
| Shoot action intercepted | Action 4 replaced with no-op (0) when ammo = 0 | `_apply_ammo_limit()` |

### Game 3 — Stamina Decay (`src/wrappers/kaz_wrapper.py`)

| Parameter | Value | Location |
|-----------|-------|----------|
| `stamina_enabled` | `True` when `game_level >= 3` | `KAZWrapper.__init__` |
| `stamina_cost_move` | 0.01 per movement action | `KAZWrapper.__init__` |
| `stamina_cost_attack` | 0.05 per sword swing | `KAZWrapper.__init__` |
| Applies to | Knights only (`agent.startswith("knight")`) | `_compute_stamina_cost()` |

---

## Execution Steps

### Step 1 — Pre-flight: Smoke-test Game Levels 2 and 3

Verify the wrapper activates correctly for each game level before committing to long training runs.

```bash
# From project root (~/ADLProject2)
./.venv/bin/python -c "
from src.wrappers.kaz_wrapper import KAZWrapper
for gl in [2, 3]:
    env = KAZWrapper(game_level=gl, vector_state=True, max_cycles=50)
    obs, info = env.reset()
    print(f'Game {gl}: agents={env.agents}, ammo_enabled={env.ammo_limit_enabled}, stamina_enabled={env.stamina_enabled}')
    for step in range(20):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        if not env.agents:
            break
    print(f'  -> Ran {step+1} steps OK. Sample reward_info: {list(infos.values())[0].get(\"reward_info\", {}) if infos else \"N/A\"}')
    env.close()
print('Pre-flight PASSED')
"
```

**Pass criteria:** No exceptions, `ammo_enabled=True` for both, `stamina_enabled=True` only for Game 3, `reward_info` dict present in infos.

---

### Step 2 — Train Game 2 (Ammo Restriction)

Train from scratch with the same hyperparameters as Game 1. The only change is `--game_level 2`.

```bash
./.venv/bin/python src/train.py \
    --game_level 2 \
    --total_timesteps 500000 \
    --lr 3e-4 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_range 0.2 \
    --n_epochs 4 \
    --batch_size 512 \
    --rollout_steps 2048 \
    --hidden_dim 256 \
    --ent_coef 0.01 \
    --vf_coef 0.5 \
    --max_grad_norm 0.5 \
    --seed 42 \
    --max_cycles 900 \
    --log_interval 5 \
    --save_interval 20
```

**Expected outputs:**
- Checkpoints in `models/game2/` (periodic + `final.pt`)
- TensorBoard logs in `results/tensorboard/game2/`
- Metrics JSON at `results/game2_baseline_metrics.json`

**What to watch during training:**
- `reward/aggression` should be lower than G1 (archers penalized for dry fires).
- Dry fire penalties should decrease over time as the archer learns ammo conservation.
- Episode length may increase (fewer kills → zombies persist longer).

---

### Step 3 — Evaluate Game 2

```bash
./.venv/bin/python src/evaluate.py \
    --checkpoint models/game2/final.pt \
    --episodes 10 \
    --seed 123
```

**Record demo episodes:**
```bash
./.venv/bin/python src/evaluate.py \
    --checkpoint models/game2/final.pt \
    --episodes 3 \
    --record
```

**Expected artifacts:**
- `results/game2_eval_results.json`
- `results/game2_demo/episode_{1,2,3}.mp4`

---

### Step 4 — Train Game 3 (Ammo + Stamina Decay)

Game 3 stacks both constraints. Train from scratch.

```bash
./.venv/bin/python src/train.py \
    --game_level 3 \
    --total_timesteps 500000 \
    --lr 3e-4 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_range 0.2 \
    --n_epochs 4 \
    --batch_size 512 \
    --rollout_steps 2048 \
    --hidden_dim 256 \
    --ent_coef 0.01 \
    --vf_coef 0.5 \
    --max_grad_norm 0.5 \
    --seed 42 \
    --max_cycles 900 \
    --log_interval 5 \
    --save_interval 20
```

**Expected outputs:**
- Checkpoints in `models/game3/` (periodic + `final.pt`)
- TensorBoard logs in `results/tensorboard/game3/`
- Metrics JSON at `results/game3_baseline_metrics.json`

**What to watch during training:**
- Knight's `reward/aggression` should drop (stamina costs deducted).
- Knight should learn to reduce unnecessary movement and swings.
- Overall team return may be lower than G2 — that is expected.

---

### Step 5 — Evaluate Game 3

```bash
./.venv/bin/python src/evaluate.py \
    --checkpoint models/game3/final.pt \
    --episodes 10 \
    --seed 123
```

**Record demo episodes:**
```bash
./.venv/bin/python src/evaluate.py \
    --checkpoint models/game3/final.pt \
    --episodes 3 \
    --record
```

**Expected artifacts:**
- `results/game3_eval_results.json`
- `results/game3_demo/episode_{1,2,3}.mp4`

---

### Step 6 — Comparative Analysis (G1 vs G2 vs G3)

After both training runs, compare the three game levels side by side.

#### 6a. TensorBoard comparison

```bash
./.venv/bin/python -m tensorboard.main \
    --logdir results/tensorboard \
    --port 6006 \
    --bind_all
```

Open TensorBoard and compare:
- `reward/aggression` curves across G1, G2, G3
- `reward/preservation` curves (should be ~0 for all three since team reward is off)
- `metrics/kill_density` — expect G1 > G2 > G3
- `metrics/episode_length` — expect G3 ≥ G2 ≥ G1

#### 6b. Build ablation comparison table

Create a script or notebook to load the three `*_eval_results.json` files and produce a summary table:

| Metric | Game 1 | Game 2 | Game 3 |
|--------|--------|--------|--------|
| Mean return | | | |
| Std return | | | |
| Mean aggression | | | |
| Mean preservation | | | |
| Mean ep length | | | |
| Kill density | | | |

Save this table to `results/phase2_ablation_table.md`.

#### 6c. Behavioral observations

Document qualitative observations in `results/phase2_observations.md`:
- **Game 2:** Does the archer stop shooting when low on ammo? Does it learn to prioritize shots?
- **Game 3:** Does the knight move less? Does it adopt a "wait and ambush" posture instead of chasing?
- **G2 vs G3:** How does the added stamina constraint change team dynamics?

---

### Step 7 — Hyperparameter Tuning (If Needed)

Only proceed here if Step 6 reveals problems (e.g., reward collapse, no behavioral change).

**Tuning levers by priority:**

1. **`max_ammo_per_archer`** (currently 15) — if archer never runs out, increase constraint by lowering to 10 or 8.
2. **`dry_fire_penalty`** (currently −0.5) — if archer doesn't learn trigger discipline, increase to −1.0.
3. **`stamina_cost_move` / `stamina_cost_attack`** (currently 0.01 / 0.05) — if knight doesn't slow down, increase to 0.03 / 0.10.
4. **`ent_coef`** (currently 0.01) — if policy collapses to a single action, increase to 0.02–0.05.
5. **`total_timesteps`** — if learning curve is still improving at 500k, extend to 750k or 1M.

After tuning, retrain and re-evaluate (repeat Steps 2–6).

---

### Step 8 — Update Project Files

After successful training and evaluation:

1. **`plan.md`** — Mark Phase 2 with ✅ and fill in key results (mean return, kill density).
2. **`README.md`** — Update the Progress table for Phase 2 row.
3. **Commit message format:** `phase2: train G2 (ammo) and G3 (stamina) — trigger discipline + economic positioning`

---

## Artifact Checklist

| Artifact | Path | Status |
|----------|------|--------|
| Game 2 checkpoint | `models/game2/final.pt` | ⬚ |
| Game 2 metrics | `results/game2_baseline_metrics.json` | ⬚ |
| Game 2 eval results | `results/game2_eval_results.json` | ⬚ |
| Game 2 demo videos | `results/game2_demo/` | ⬚ |
| Game 2 TensorBoard logs | `results/tensorboard/game2/` | ⬚ |
| Game 3 checkpoint | `models/game3/final.pt` | ⬚ |
| Game 3 metrics | `results/game3_baseline_metrics.json` | ⬚ |
| Game 3 eval results | `results/game3_eval_results.json` | ⬚ |
| Game 3 demo videos | `results/game3_demo/` | ⬚ |
| Game 3 TensorBoard logs | `results/tensorboard/game3/` | ⬚ |
| Phase 2 ablation table | `results/phase2_ablation_table.md` | ⬚ |
| Phase 2 observations | `results/phase2_observations.md` | ⬚ |
| `plan.md` updated | `plan.md` | ⬚ |
| `README.md` updated | `README.md` | ⬚ |
