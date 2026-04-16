# V2 Implementation Plan: Death Penalty Reward Signal

> **Context:** Elizabeth identified that G2/G3 agents learn passivity because there are
> penalties for acting (ammo waste, stamina cost) but zero cost for dying. This plan adds
> a configurable death penalty to the reward structure and retrains all 5 games.

---

## 1. Design Decision

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `death_penalty` | **2.0** | ≈ 2 kills of negative reward. High enough that "die passively" (net −2.0) is clearly worse than "kill 3, then die" (net +1.0). Low enough not to overwhelm G1's ~5.8 kills/ep. |
| Applied to | **All games (G1–G5)** | Agent death always has a cost in real domains. Not a "constraint" — a base survival signal. G1 remains easiest (kill reward + death penalty, no ammo/stamina). |
| KAZWrapper default | `0.0` | Backward compat with existing V1 checkpoints. `train.py` CLI default is `2.0`. |

### Alternative considered: per-step survival bonus

A small positive reward per step alive (+0.01/step) was considered. **Rejected because:**
- Over 900 steps, bonus = 9.0 — would dominate kill reward (1.0/kill)
- Harder to tune (too high → pure hiding, too low → no effect)
- Death penalty is simpler to explain: "dying costs 2 kills"

### Key experiment

**Does death penalty alone fix deterministic passivity in G3 (still MLP, no GRU)?**
- If yes → death penalty was the missing piece; GRU was an architectural workaround
- If no → GRU temporal memory was genuinely necessary; death penalty is complementary
- Either answer strengthens the presentation

---

## 2. Code Changes

### 2.1 `src/wrappers/kaz_wrapper.py`

**A) Add `death_penalty` parameter to `__init__`:**
- New param in signature: `death_penalty: float = 0.0` (after `seed`)
- Store: `self.death_penalty = death_penalty`

**B) Apply in `step()` — inside the `for agent in raw_rewards:` loop:**

Insert AFTER `total_reward` is computed (after the weighted combination line)
and BEFORE `shaped_rewards[agent] = total_reward`:

```python
            # Death penalty — applied when agent is terminated
            death_cost = 0.0
            if self.death_penalty > 0 and terminations.get(agent, False):
                death_cost = self.death_penalty
                total_reward -= death_cost
```

**C) Add `death_penalty` to `reward_info` dict:**

Update the existing `infos[agent]["reward_info"]` dict to include:
```python
            infos[agent]["reward_info"] = {
                "raw_kill": float(raw_rewards[agent]),
                "aggression": reward_aggression,
                "preservation": reward_preservation,
                "death_penalty": -death_cost,  # NEW — negative when agent dies
                "total": total_reward,
            }
```

### 2.2 `src/train.py`

**A) Add CLI argument** (in `parse_args()`, after `--max_cycles`):
```python
    p.add_argument("--death_penalty", type=float, default=2.0,
                    help="Penalty applied when an agent dies (0 = disabled)")
```

**B) Pass to KAZWrapper** (in `train()`, the `KAZWrapper(...)` constructor call around line 177):
```python
    env = KAZWrapper(
        game_level=args.game_level,
        max_cycles=args.max_cycles,
        vector_state=True,
        seed=args.seed,
        death_penalty=args.death_penalty,  # NEW
    )
```

**C) Add death penalty tracking to RolloutBuffer.__init__** (around line 83):
```python
        self.death_penalty_rewards = []  # NEW
```

**D) Add to RolloutBuffer.add()** (around line 90):
- Add `death_penalty=0.0` parameter
- Add `self.death_penalty_rewards.append(death_penalty)` in the body

**E) Track per-episode death penalty** (around line 238, with the other ep_ trackers):
```python
    ep_death_penalty = defaultdict(float)  # NEW
```

**F) Accumulate death penalty in the reward loop** (around line 342, inside `for agent in rewards:`):
```python
                dp = ri.get("death_penalty", 0.0)
```
Then after line 354 (where buffer indices are assigned):
```python
                buffer.death_penalty_rewards[buf_idx] = dp
```
And after line 359 (where ep_ accumulators are updated):
```python
                ep_death_penalty[agent] += dp
```

**G) Reset ep_death_penalty on episode boundary** (around line 276, where other ep_ dicts reset):
```python
                ep_death_penalty = defaultdict(float)  # NEW — add with the other resets
```

**H) Add to rollout episode summary** (around line 268, add a new list):
```python
        rollout_death = []  # NEW — add near other rollout_ lists (line ~251)
```
And in the episode boundary block (around line 270):
```python
                rollout_death.append(
                    float(np.mean(list(ep_death_penalty.values()))) if ep_death_penalty else 0.0
                )
```

**I) TensorBoard logging** (around line 464, after preservation logging):
```python
            avg_death_penalty = float(np.mean(buffer.death_penalty_rewards))
            writer.add_scalar("reward/death_penalty", avg_death_penalty, global_step)
```

**J) Print death penalty in startup banner** (after line 174):
```python
    print(f"[train] Death penalty: {args.death_penalty}")
```

### 2.3 `src/evaluate.py`

**A) Read death_penalty from checkpoint** (around line 53, with other train_args reads):
```python
    death_penalty = train_args.get("death_penalty", 0.0)
```

**B) Pass to KAZWrapper** (around line 64, the KAZWrapper constructor):
```python
    env = KAZWrapper(
        game_level=game_level,
        vector_state=True,
        render_mode=render_mode,
        seed=args.seed,
        death_penalty=death_penalty,  # NEW
    )
```

**C) Track death penalty in episode loop** (around line 104, add with other ep_ dicts):
```python
        ep_death = defaultdict(float)  # NEW
```

**D) Accumulate in reward loop** (around line 158, after ep_pres):
```python
                ep_death[agent] += ri.get("death_penalty", 0.0)
```

**E) Add to results JSON** (around line 216, with other metrics):
```python
        "mean_death_penalty": float(np.mean(all_death)),
```
(Also need `all_death = []` list around line 96 and `all_death.append(...)` around line 169.)

**F) Print death penalty info** (around line 55, after device info):
```python
    print(f"[eval] Death penalty: {death_penalty}")
```

### 2.4 `src/phase5.py` (optional — consistency only)

Phase 5 kill counting already correctly uses `infos["raw_kill"]`, NOT shaped rewards.
No functional bug. But for consistency, the KAZWrapper constructors should receive the
death_penalty from the checkpoint:

**A) In `run_episodes()` (line 71)** — read death_penalty from the caller or default 0.0:
- Add `death_penalty=0.0` parameter to `run_episodes()` signature
- Pass to `KAZWrapper(..., death_penalty=death_penalty)`

**B) In `part2_collapse_graph()` (line 199-202)** — pass death_penalty from loaded checkpoints:
```python
    dp_g1 = ckpt_g1_args.get("death_penalty", 0.0)
    dp_g5 = ckpt_g5_args.get("death_penalty", 0.0)
```
Then pass to `run_episodes(..., death_penalty=dp_g1)` and `run_episodes(..., death_penalty=dp_g5)`.

**C) In `part4_sidebyside()` (lines 456-459)** — pass death_penalty to KAZWrapper constructors.

**D) In `load_net()` (line 41)** — cache death_penalty from checkpoint args in the returned tuple
for downstream use by callers.

> **Note:** These phase5.py changes don't affect any output metrics (kill counts, saliency,
> video frames). They only affect the internal shaped reward values, which are not displayed
> or recorded. Implement them for code hygiene but don't block on them.

### 2.5 No changes needed

- `src/models/mappo_net.py` — model architecture unchanged
- `src/utils.py` — device detection unchanged
- `src/test_env.py` — uses default death_penalty=0.0, fine for smoke test
- `src/wrappers/__init__.py` — no changes

---

## 3. Training (USER runs)

### 3.1 Back up existing checkpoints

```bash
cp -r models/ models_v1_backup/
```

### 3.2 Training commands

Run these **sequentially**. Each game's checkpoint is needed before the next
(G5 transfers from G4). Total time: ~2.5 hours on RTX 5070 Ti.

```bash
# G1 — MLP, kill reward + death penalty (~20 min)
./.venv/bin/python src/train.py --game_level 1 --total_timesteps 500000 --death_penalty 2.0

# G2 — MLP + ammo limit + death penalty (~20 min)
./.venv/bin/python src/train.py --game_level 2 --total_timesteps 500000 --death_penalty 2.0

# G3 — MLP + ammo + stamina + death penalty (~20 min)
./.venv/bin/python src/train.py --game_level 3 --total_timesteps 500000 --death_penalty 2.0

# G4 — Attention + team reward + death penalty (~40 min)
./.venv/bin/python src/train.py --game_level 4 --arch attention --total_timesteps 1000000 --death_penalty 2.0

# G5 — Attention+GRU + fog + death penalty, transfer from NEW G4 (~60 min)
./.venv/bin/python src/train.py \
    --game_level 5 --arch attention_gru \
    --transfer_from models/game4/final.pt \
    --total_timesteps 1500000 --max_cycles 900 \
    --death_penalty 2.0
```

### 3.3 What to watch during training

- **G3 is the key experiment.** Check TensorBoard: if attack actions increase
  over training, death penalty is working. If G3 det. attack rate > 0% in eval,
  death penalty alone partially fixes passivity.
- **G1 mean return will be lower** (death penalty subtracts from total) but
  kill density should remain similar (~0.01).
- **Entropy collapse:** If any game shows entropy → 0 quickly, re-run with
  `--min_entropy 0.3` appended to the command.
- **Death penalty TensorBoard scalar** (`reward/death_penalty`): should show
  the penalty becoming less frequent over training (agents learn to survive longer).

---

## 4. Evaluation Pipeline (CASCADE runs after ALL training completes)

### 4.1 Stochastic evaluation (all 5 games)

```bash
./.venv/bin/python src/evaluate.py --checkpoint models/game1/final.pt --episodes 10
./.venv/bin/python src/evaluate.py --checkpoint models/game2/final.pt --episodes 10
./.venv/bin/python src/evaluate.py --checkpoint models/game3/final.pt --episodes 10
./.venv/bin/python src/evaluate.py --checkpoint models/game4/final.pt --episodes 10
./.venv/bin/python src/evaluate.py --checkpoint models/game5/final.pt --episodes 10
```

### 4.2 Deterministic evaluation (all 5 games)

```bash
./.venv/bin/python src/evaluate.py --checkpoint models/game1/final.pt --episodes 10 --deterministic
./.venv/bin/python src/evaluate.py --checkpoint models/game2/final.pt --episodes 10 --deterministic
./.venv/bin/python src/evaluate.py --checkpoint models/game3/final.pt --episodes 10 --deterministic
./.venv/bin/python src/evaluate.py --checkpoint models/game4/final.pt --episodes 10 --deterministic
./.venv/bin/python src/evaluate.py --checkpoint models/game5/final.pt --episodes 10 --deterministic
```

### 4.3 Demo recording

```bash
./.venv/bin/python src/evaluate.py --checkpoint models/game1/final.pt --episodes 3 --record
./.venv/bin/python src/evaluate.py --checkpoint models/game5/final.pt --episodes 3 --record
```

### 4.4 Phase 5 artifact regeneration

```bash
./.venv/bin/python src/phase5.py
```

This regenerates all 4 artifacts:
- `results/kill_density_evolution.png`
- `results/collapse_graph.png`
- `results/saliency_comparison.png`
- `results/demo_sidebyside.mp4`

---

## 5. Documentation Updates (CASCADE does after evaluation)

### 5.1 `results/final_ablation_table.md`
- Regenerate with new numbers from eval JSONs
- Add "Death Penalty" row to the constraint stack section
- Compare V1 vs V2 numbers if interesting

### 5.2 `README.md`
- Update "Results at a Glance" table with new numbers
- Update "Constraint Stack" table — add death penalty row (all games)
- Add note about V2 design change in "What This Project Does"
- Update "Progress" table with new key results
- Update training commands to include `--death_penalty 2.0`

### 5.3 `docs/report.tex` + `docs/report.pdf`
- Update ablation table (Table 1) with new numbers
- Add `death_penalty` to hyperparameters table (Table 2)
- Update results and discussion sections
- Add paragraph about design iteration (Elizabeth's insight → death penalty → retrain)
- Recompile PDF: `cd docs && pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex`

### 5.4 `docs/presentation_guide.md`
- Update all numbers throughout
- Add slide about death penalty design decision and its impact
- Update the "Q&A preparation" section with the death penalty rationale

### 5.5 `results/phase4_observations.md`
- Update with new G5 observations if behavior changes significantly
- Note whether deterministic passivity is still the central finding or if death penalty shifts the narrative

### 5.6 `plan.md`
- Add a "V2: Death Penalty" section documenting the design iteration

---

## 6. Final Commit & Push

```bash
git add -A
git commit -m "v2: add death penalty (2.0) + full retrain G1-G5 + re-evaluation

Motivation: G2/G3 passivity caused by zero cost of dying.
Death penalty makes 'die passively' (net -2.0) clearly worse than
'kill and survive' — addresses the survival incentive gap.

Code changes:
- kaz_wrapper.py: death_penalty param, applied on termination
- train.py: --death_penalty CLI arg (default 2.0), TensorBoard logging
- evaluate.py: reads death_penalty from checkpoint, passes to wrapper
- phase5.py: passes death_penalty to wrapper constructors

Training: G1-G5 retrained with death_penalty=2.0
Results: [INSERT KEY NUMBERS]"

git push origin v2-death-penalty
```

Then merge to main:
```bash
git checkout main
git merge v2-death-penalty
git push origin main
```

---

## 7. V1 Reference Numbers (for comparison)

| Game | Label | Mean Return | Raw Kill Density | Det. Attack % |
|------|-------|-------------|-----------------|---------------|
| G1 | Greedy Soldier | 5.80 ± 2.94 | 0.01089 | ~100% |
| G2 | Risk Avoider | 0.30 ± 0.39 | 0.00431 | <5% |
| G3 | Fully Passive | 0.15 ± 0.24 | 0.00279 | ~0% |
| G4 | Recovering Cooperator | 0.42 ± 0.38 | 0.00361 | ~0% |
| G5 | Fire Discipline | 0.38 ± 0.31 | 0.00299 | 69.9% |

> Note: V2 "mean return" will be lower across the board (death penalty subtracts
> from total reward). Compare kill density and det. attack rate, not raw returns.
