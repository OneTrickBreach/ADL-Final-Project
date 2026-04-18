# Design Specification: Structured Teamwork in Multi-Agent PPO

> Reproducible design and evaluation spec for the 7-game ablation on
> PettingZoo Knights-Archers-Zombies. Academic write-up is in
> `docs/report.pdf`; milestone ledger in `plan.md`.

---

## 1. Research question

*How much cooperative performance comes from building coordination into the
environment itself, rather than shaping the reward?*

We isolate four coordination primitives and measure each one's contribution:

1. **Ammo / stamina discipline** — per-archer / shared ammo pool + per-knight
   stamina pool.
2. **Role assignment** — one forward knight, one end-zone patrol knight; role is
   observed by the policy as a feature.
3. **Target locks** — knights lock the nearest zombie inside a radius; archers
   skip knight-locked zombies.
4. **Pragmatic override** — archer overrides a knight lock if the zombie will
   cross the end line before the knight can intercept.

---

## 2. Environment and evaluation protocol

| Parameter                          | Value |
|------------------------------------|-------|
| PettingZoo KAZ                     | v10, `vector_state=True`, `line_death=False` |
| Agents                             | 2 archers + 2 knights |
| Episode length                     | 30 s @ 15 FPS = **450 steps** |
| Score                              | `kills − failures` per episode |
| Failure definition (see §3)        | line-cross OR agent-breach (zombie contact) |
| Evaluation seed convention         | episode *i* uses seed `42 + i` identically across all games |
| Episodes per game                  | 10 stochastic + 10 deterministic (G2–G5 only) |

### 2.1 Environment pressure parameters

| Parameter                          | Value |
|------------------------------------|------:|
| `spawn_rate`                       | 8     (KAZ default 20; lower = denser) |
| `max_zombies`                      | 20    (KAZ default 10) |
| `knight_lock_radius_fraction`      | 0.25  (320 px on a 1280-wide screen) |
| `global_ammo_pool`                 | 60    |
| `individual_ammo_pool`             | 30    |
| `knight_stamina_pool`              | 150   |
| `end_zone_fraction`                | 0.20  (bottom 20 % of playfield) |

These pressure values were chosen so that (a) the failure penalty is a live
training signal (`failures_mean > 0` under every heuristic baseline) and
(b) ammo / stamina are depleted to ~85–97 % per episode, keeping discipline
meaningful.

---

## 3. Failure detector

A failure is logged per step from two additive sources:

### 3.1 Line-cross

A zombie in the *current* `zombie_list` whose `rect.centery` exceeds
`SCREEN_HEIGHT − ZOMBIE_Y_SPEED` is counted as a failure on the first step it
crosses the threshold. Each zombie's Python `id()` is cached in a
per-episode set so multi-step lingering doesn't double-count.

KAZ does not remove crossed zombies from `zombie_list` (they only get removed
when killed by arrow or sword), so position-based detection on the live list
is the correct signal.

### 3.2 Agent-breach

The size of `archer_list + knight_list` is monitored per step. Any decrease is
attributed to zombie contact — the only code path in KAZ that removes agents
from those lists is `zombie_hit_archer` / `zombit_hit_knight`.

### 3.3 Episode-continuation override

When `self._env.unwrapped.run` is flipped to `False` due to a zombie cross but
agents are still alive and the natural truncation horizon has not been reached,
the wrapper flips `run` back to `True` and removes the already-counted crossed
zombie. This lets a single rollout accumulate multiple failures, which produces
a richer training signal than the KAZ default (which ends the episode on the
first cross).

### 3.4 Per-step penalty cap

To prevent catastrophic advantage spikes when several failures stack on the
same step, the reward term is clamped:
`penalty = −1.0 × min(failures_this_step, 3)`
applied to all living agents. The raw counter (used for metrics) is left
unclamped.

---

## 4. Observation and reward

### 4.1 Observation augmentation (140 dimensions)

Each agent's KAZ observation `(27, 5)` is flattened to 135 dimensions and
concatenated with a 5-dim extras vector:

| Index | Feature          | Range |
|-------|------------------|-------|
| 0     | `role_id`        | 0 archer / 1 forward-knight / 2 patrol-knight |
| 1     | `in_end_zone`    | {0, 1} |
| 2     | `ammo_frac`      | [0, 1] |
| 3     | `stamina_frac`   | [0, 1] |
| 4     | `has_lock`       | {0, 1} |

Total `obs_dim = 140`, identical across all seven games. This preserves
checkpoint compatibility for transfer-learning between adjacent games.

### 4.2 Reward composition

```
r_i = r_kill_i                                          (KAZ's +1 per elimination)
    + b_r · 1{roles_on} · φ_r(i)                        (role bonus, b_r = 1e-3)
    + b_ℓ · 1{lock_on} · φ_ℓ(i)                         (lock bonus, b_ℓ = 2e-3)
    − p_f · min(failures_this_step, 3)                  (failure penalty, p_f = 1.0)
    − p_m · 1{invalid_action}(a_i)                      (mask penalty, p_m = 0.05)
```

- `φ_r(i) = 1` iff agent *i* is in its assigned zone.
- `φ_ℓ(i) = 1` iff agent *i* attacked while holding a valid lock.
- Failures are shared (every living agent pays the cost of a breach).

### 4.3 Action masking

Invalid actions (e.g. firing with empty ammo, patrol-knight leaving its band
when `lock_on` is true) are subject to:

- **Soft mask** (default): the invalid action executes as a no-op and the
  agent receives `−p_m` so gradients flow through the choice.
- **Hard mask** (patrol-knight boundary crossings when `lock_on` is true): the
  action is remapped to no-op without penalty; prevents the wrapper-enforced
  role violation from being executed at all.

---

## 5. Architecture

Shared-parameter MAPPO across all 4 agents (role is a feature, not a separate
network).

```
obs(140) ──split──► entities(27×5) ─► MHA-encoder ─► Pool ─► z_t
                │                                           │
                └── extras(5) ────── Linear(5 → H) ──────── + ─► (GRUCell for G3–G5) ─► MLP ─► π, V
```

- G2: Attention only (432,265 params)
- G3–G5: Attention + GRUCell (827,017 params)
- Transfer-learning: G3 copies 28 / 32 tensors from G2 (the 4 GRU parameter
  tensors are new); G4 and G5 each copy 32 / 32 from their predecessor.

Hyper-parameters: lr 3 × 10⁻⁴, γ = 0.99, λ_GAE = 0.95, clip ε = 0.2, entropy
β = 0.01, rollout 2048 steps, mini-batch 512, 4 PPO epochs, ‖∇θ‖ ≤ 0.5.

---

## 6. Training orchestration

`scripts/mega_train_v3_1.sh` runs end-to-end:

1. Evaluate G0 / G1a / G1b heuristics (10 episodes, seed 42).
2. Auto-select ammo mode: if `G1a.score_mean ≥ G1b.score_mean` → `global`
   else `individual`. Writes the choice to `results/v3_1/ammo_mode.txt`.
3. Train G2 from scratch, 1 M steps.
4. Train G3 transfer-from-G2, 1.5 M steps.
5. Train G4 transfer-from-G3, 1.5 M steps.
6. Train G5 transfer-from-G4, 2 M steps.

Total ≈ 6 M timesteps, ~3.5 h wall-clock on an RTX 5070 Ti.

All outputs: `models/v3_1/{g2..g5}/final.pt`, `results/v3_1/`,
`results/tensorboard_v3_1/`.

---

## 7. Ship-gates (must all pass)

1. `g1a / g1b failures_mean > 0` — proves the heuristic baselines are under
   real pressure.
2. `failures_mean > 0` in at least one learned game — proves the penalty is
   active at training time.
3. `G3 ≥ G2` (stochastic) — roles add value.
4. `G4 ≥ G3` (stochastic) — locks add value.
5. `G5 ≥ G4` (stochastic) — pragmatic override adds value.

*Measured values (all pass):* G1a failures = 2.9, G1b failures = 2.9, learned
failures ∈ [1.8, 2.8], G3 − G2 = +2.30, G4 − G3 = +3.30, G5 − G4 = +3.80.

---

## 8. Evaluation pipeline

```bash
RESULTS=results/v3_1

# Heuristic baselines
for g in g0 g1a g1b; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --episodes 10 --seed 42 --results_dir $RESULTS
done

# Learned: stochastic + deterministic
for g in g2 g3 g4 g5; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3_1/$g/final.pt \
      --episodes 10 --seed 42 --results_dir $RESULTS
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3_1/$g/final.pt \
      --episodes 10 --seed 42 --deterministic --results_dir $RESULTS
done

# Demo MP4s (10 eps × 7 games = 70 recordings)
for g in g0 g1a g1b; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --episodes 10 --seed 42 \
      --record --results_dir $RESULTS --output_suffix record
done
for g in g2 g3 g4 g5; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3_1/$g/final.pt \
      --episodes 10 --seed 42 --record --results_dir $RESULTS --output_suffix record
done

# Artifacts: figures, side-by-side mp4, markdown ablation table
./.venv/bin/python src/phase_artifacts_v3.py \
    --results_dir $RESULTS --output_dir $RESULTS --models_dir models/v3_1
```

---

## 9. Output schema

```
results/v3_1/
├── ammo_mode.txt                        # "global" or "individual"
├── ablation_table.md                    # auto-generated markdown
├── score_evolution.png                  # 7-bar chart
├── metric_breakdown.png                 # 2×2 grid: %ammo, %stamina, kills A/K, attacks A/K
├── saliency_v3.png                      # G3 vs G5 attention heatmaps
├── demo_sidebyside_v3.mp4               # G0 heuristic left, G5 learned right, seed 42
├── g{0,1a,1b,2,3,4,5}_eval_results.json
├── g{2,3,4,5}_eval_results_det.json
├── g{0..5}_eval_results_record.json     # re-run alongside video recording
└── g{0..5}_demo/episode_{1..10}_seed{42..51}.mp4
```

Every JSON contains: `score_mean/std`, `kills_archer_mean`, `kills_knight_mean`,
`attacks_archer_mean`, `attacks_knight_mean`, `failures_mean`,
`ammo_pct_expended_mean`, `stamina_pct_expended_mean`, `episode_length_mean`,
plus per-agent breakdowns.

---

## 10. Non-goals

- No inter-agent communication channel. The coordination signal flows through
  structural primitives alone.
- No centralised critic. MAPPO with shared parameters is the only training
  algorithm.
- No reward re-weighting between games (only the feature flags `roles_on` /
  `lock_on` / `pragmatic_on` change).
- No observation-dim change between games. `obs_dim = 140` for all seven.
