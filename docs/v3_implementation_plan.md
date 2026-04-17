# V3 Implementation Plan: Full Environment Overhaul

> **Source of truth:** `docs/v3_psudoplan.txt` (group-approved design)
> **Branch:** `v3-overhaul`
> **Scope:** Complete redesign of game mechanics, ablation narrative, and evaluation pipeline.
> V1 and V2 codebases are preserved as reference — nothing old is modified.

---

## 1. Philosophy & Scope

V1/V2 told the story "constraints collapse behavior; GRU recovers it." V3 tells a cleaner
story: **structured teamwork with progressive coordination primitives** (ammo discipline
→ role assignment → target locks → pragmatic override).

**Seven games, three learning levels:**

| Game | Name | Learning? | Architecture | What's new |
|------|------|-----------|--------------|-----------|
| G0 | Unrestricted | ❌ heuristic | — | Baseline: no ammo/stamina caps |
| G1a | Global-pool ammo | ❌ heuristic | — | Shared ammo pool + knight stamina; archers immobile |
| G1b | Individual-pool ammo | ❌ heuristic | — | Per-archer ammo pool + knight stamina; archers immobile |
| G2 | Basic teamwork | ✅ trained | Attention | Best ammo mode from G1a/G1b; learn sniper/hunter roles |
| G3 | Role assignment | ✅ trained | Attention + GRU | 1 patrol knight (end zone), 1 forward knight |
| G4 | Target locks | ✅ trained | Attention + GRU | Lock radius for knights; archer skips knight-locked |
| G5 | Pragmatic override | ✅ trained | Attention + GRU | Archer overrides knight lock if zombie will cross first |

---

## 2. Concrete Assumptions (commit unless overridden)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_archers`, `num_knights` | 2, 2 | Pseudoplan §2 |
| `duration_seconds` | 30 | Pseudoplan §3 |
| `max_cycles` | **450** | 30s × 15 FPS (verified from KAZ `const.FPS`) |
| `global_ammo_pool` | **30** arrows | Clean divisor for 2 archers |
| `individual_ammo_pool` | **15** arrows/archer | `ceil(30/2) = 15` |
| `knight_stamina_pool` | **100** actions | Roughly enough for 22% of the episode |
| `end_zone_fraction` | **0.20** | Bottom 20% of screen (§7d) |
| `knight_lock_radius_fraction` | **0.15** | 15% of width = 192px (§7e) |
| `spawn_rate` / `max_zombies` | **20 / 10** | KAZ defaults, fixed across all V3 games |
| Evaluation seed | **42** (fixed) | Identical zombie pattern across games |
| KAZ flags | `line_death=False, killable_archers=True, killable_knights=True, vector_state=True, pad_observation=True` | Needed for failure counting |

**Training budgets:**

| Game | Total timesteps | Wall-clock @ ~500 SPS |
|------|----------------|------------------------|
| G2 (attention, scratch) | 1,000,000 | ~35 min |
| G3 (attn+GRU, ← G2) | 1,500,000 | ~50 min |
| G4 (attn+GRU, ← G3) | 1,500,000 | ~50 min |
| G5 (attn+GRU, ← G4) | 1,500,000 | ~50 min |
| **Total** | **5.5M** | **~3 hours** |

---

## 3. File Structure

**Preserved unchanged (V1/V2 reference):**
- `src/wrappers/kaz_wrapper.py`, `src/train.py`, `src/evaluate.py`, `src/phase5.py`

**New files:**
```
src/
├── wrappers/kaz_wrapper_v3.py    # NEW
├── policies/__init__.py          # NEW
├── policies/heuristic.py         # NEW
├── models/mappo_net.py           # MODIFIED (add extra_dim param)
├── train_v3.py                   # NEW
├── evaluate_v3.py                # NEW
└── phase_artifacts_v3.py         # NEW
scripts/mega_train_v3.sh          # NEW
models/v3/{g2,g3,g4,g5}/final.pt  # training outputs
results/v3/                       # eval outputs, figures, ablation table
results/tensorboard_v3/           # training curves
```

---

## 4. Code Specifications

### 4.1 `src/wrappers/kaz_wrapper_v3.py` (NEW)

**Class `KAZWrapperV3` — constructor signature:**
```python
def __init__(
    self,
    game_level: str,           # "g0"|"g1a"|"g1b"|"g2"|"g3"|"g4"|"g5"
    duration_seconds: int = 30,
    num_archers: int = 2, num_knights: int = 2,
    global_ammo_pool: int = 30, individual_ammo_pool: int = 15,
    knight_stamina_pool: int = 100,
    end_zone_fraction: float = 0.20,
    knight_lock_radius_fraction: float = 0.15,
    spawn_rate: int = 20, max_zombies: int = 10,
    ammo_mode_override: str | None = None,  # "global"|"individual" — required for G2+
    vector_state: bool = True, render_mode: str | None = None, seed: int | None = None,
):
```

**Game config table (internal):**
```python
GAME_CONFIGS = {
  "g0":  dict(ammo_on=False, stamina_on=False, roles_on=False, lock_on=False, pragmatic=False, archer_immobile=False),
  "g1a": dict(ammo_on=True,  ammo_mode="global",     stamina_on=True,  archer_immobile=True,  roles_on=False, lock_on=False, pragmatic=False),
  "g1b": dict(ammo_on=True,  ammo_mode="individual", stamina_on=True,  archer_immobile=True,  roles_on=False, lock_on=False, pragmatic=False),
  "g2":  dict(ammo_on=True,  ammo_mode="<override>", stamina_on=True,  archer_immobile=False, roles_on=False, lock_on=False, pragmatic=False),
  "g3":  dict(ammo_on=True,  ammo_mode="<override>", stamina_on=True,  archer_immobile=False, roles_on=True,  lock_on=False, pragmatic=False),
  "g4":  dict(ammo_on=True,  ammo_mode="<override>", stamina_on=True,  archer_immobile=False, roles_on=True,  lock_on=True,  pragmatic=False),
  "g5":  dict(ammo_on=True,  ammo_mode="<override>", stamina_on=True,  archer_immobile=False, roles_on=True,  lock_on=True,  pragmatic=True),
}
```

**Runtime state (reset each episode):**
```python
self.cycle = 0
self.global_ammo = global_ammo_pool
self.archer_ammo = {a: individual_ammo_pool for a in archers}
self.knight_stamina = {k: knight_stamina_pool for k in knights}
self.knight_role = {knights[0]: "forward", knights[1]: "patrol"}   # fixed assignment
self.lock_target = {a: None for a in all_agents}   # id(zombie) or None
self.failure_count = 0
self.zombie_prev = {}        # id(z) -> (x, y)
self.attack_count = {a: 0 for a in all_agents}
self.kill_count = {a: 0 for a in all_agents}
# fetched once at wrapper init (constants from KAZ):
self.ZOMBIE_Y_SPEED = getattr(kaz_const, "ZOMBIE_Y_SPEED", 5)
self.KNIGHT_SPEED   = getattr(kaz_const, "KNIGHT_Y_SPEED", 4)
self.SCREEN_W, self.SCREEN_H = 1280, 720
self.LOCK_RADIUS_PX = int(knight_lock_radius_fraction * self.SCREEN_W)  # 192
self.END_ZONE_Y     = int((1 - end_zone_fraction) * self.SCREEN_H)     # 576
```

**Constants:**
```python
ACTION_NOOP   = 0
ACTION_ATTACK = 4   # verified in existing wrapper: external 4 = fire/attack
ACTION_MOVEMENT_SET = {1, 2, 3, 5}   # confirmed via pre-flight check §9
SCREEN_W, SCREEN_H = 1280, 720
```

**Action-masking policy (affects §4.1.5 steps 2.b–2.f):**
- **Default: soft penalty** — invalid action is still forwarded to env but the agent receives
  `-0.05` this step (and the env may or may not execute it meaningfully).
- **Fallback: hard mask** — force invalid action to `ACTION_NOOP`.
- Per Risk #3, implement BOTH modes via a constructor flag `action_mask_mode: str = "soft"`.
  Start training with soft; if policy fails to learn lock compliance, re-run with hard.
- The pseudoplan's phrasing ("cannot move out of the end zone", "will ignore") reads as
  hard constraints, but soft shaping during training is a well-known stability technique.

**`step()` procedure (authoritative ordering):**

```
1. modified = dict(actions)
2. For each agent:
   a. attack_count[a] += 1 if modified[a] == ACTION_ATTACK
   b. Archer-immobile (G1a/G1b): if archer and modified[a] in MOVEMENT → NOOP
   c. Patrol-role constraint (G4+ only): if patrol knight and movement would exit
      end zone → NOOP. (G3 uses soft reward; see §4.1.6)
   d. Ammo check: if archer attacking:
        - global mode: if global_ammo>0: global_ammo-=1 else NOOP
        - individual:  if archer_ammo[a]>0: archer_ammo[a]-=1 else NOOP
   e. Stamina check: if knight and action != NOOP:
        if stamina_on and knight_stamina[k]>0: knight_stamina[k]-=1 else NOOP
   f. Target-lock mask (G4+):
        - Archer ACTION_ATTACK with no lock → NOOP
        - Knight ACTION_ATTACK with target out of range → NOOP
3. Snapshot zombie_prev = {id(z): (z.rect.centerx, z.rect.centery) for z in raw.zombie_list}
4. obs, raw_rewards, terms, truncs, infos = self._env.step(modified)
5. Kill attribution: for each agent: if raw_rewards[a] > 0.5: kill_count[a] += 1
6. Cross detection (KAZ removes zombies that cross; we detect from snapshot):
     current_ids = {id(z) for z in raw.zombie_list}
     any_kill_this_step = any(r > 0.5 for r in raw_rewards.values())
     for zid, (x, y) in self.zombie_prev.items():
         if zid not in current_ids:
             # Disambiguate killed vs crossed: zombies cross when y > SCREEN_H - ZOMBIE_Y_SPEED.
             # If this zombie was near the bottom AND no kill reward was issued to any agent
             # this step, treat it as a crossing.  If a kill reward was issued, prefer the
             # kill attribution (the killed zombie disappears with a reward, so skip crossing count
             # for this disappearance if the zombie was NOT near the bottom threshold).
             if y >= (SCREEN_H - ZOMBIE_Y_SPEED - 5):
                 self.failure_count += 1
                 failures_this_step += 1
7. Target-lock update (G4+):
     - Invalidate dead/missing-target locks
     - For unlocked agents: acquire nearest eligible zombie per rules (§4.1.8)
8. Compute rewards (§4.1.6)
9. Augment observations (§4.1.7)
10. Truncate if self.cycle >= max_cycles; self.cycle += 1
11. return obs_aug, shaped_rewards, terms, truncs, infos
```

### 4.1.6 Reward design

Base: `raw_kill_reward` (+1/kill, from KAZ).

V3 additions (per agent, per step):
- **Failure penalty (shared):** `-1.0 × failures_this_step` applied to all living agents
- **Role bonus (G3+ only):** patrol knight: `+0.001 × in_end_zone`; forward knight: `+0.001 × (1 − in_end_zone)`
- **Lock-compliance bonus (G4+):** `+0.002` if agent respected lock rules this step

No death penalty, no 60/40 team blend (replaced by shared failure signal).

`reward_info` dict (TB + JSON):
```python
{"raw_kill":..., "role_bonus":..., "lock_bonus":..., "failure_shared":..., "total":...}
```

### 4.1.7 Observation augmentation

Base KAZ vector: `(27, 5) = 135 dims`. Append **5 extras** per agent:

| Idx | Feature | Range | Notes |
|-----|---------|-------|-------|
| 0 | `role_id` | {0,1} | 0=forward, 1=patrol; 0 if roles_on=False |
| 1 | `in_end_zone` | {0,1} | 1 if agent's y ≥ 0.80×SCREEN_H |
| 2 | `ammo_frac` | [0,1] | archers: remaining/pool; knights: 0 |
| 3 | `stamina_frac` | [0,1] | knights: remaining/pool; archers: 0 |
| 4 | `has_lock` | {0,1} | agent has a locked target |

**Final `obs_dim = 140`.** Same across G2–G5 (extras zero-filled where inapplicable) → transfer works.

### 4.1.8 Target lock acquisition (G4+)

```python
def acquire_lock(agent, zombie_map, cfg):
    """Return id(zombie) of best target, or None."""
    if is_archer(agent):
        candidates = [z for z in zombies if id(z) not in knight_locked_ids]
        # G5 pragmatic override: include knight-locked zombies that will cross first
        if cfg["pragmatic"]:
            for z, knight in knight_locks.items():
                if zombie_cross_time(z) < knight_intercept_time(knight, z):
                    candidates.append(z)
        return id(nearest(candidates, agent))  # or None if empty
    elif is_knight(agent):
        in_range = [z for z in zombies if dist(z, agent) <= LOCK_RADIUS_PX]
        if agent is patrol: in_range = [z for z in in_range if z.rect.centery >= END_ZONE_Y]
        return id(nearest(in_range, agent))
```

Where:
- `LOCK_RADIUS_PX = 0.15 × SCREEN_W = 192`
- `END_ZONE_Y = 0.80 × SCREEN_H = 576`
- `zombie_cross_time(z) = (SCREEN_H - z.rect.centery) / ZOMBIE_Y_SPEED`  (need `const.ZOMBIE_Y_SPEED`)
- `knight_intercept_time(k, z) = dist(k, z) / KNIGHT_SPEED`  (need `const.KNIGHT_SPEED`; approximate 4 px/step if unavailable)

Fetch speeds once at wrapper init from `const` module (fallback constants OK).

### 4.2 `src/policies/heuristic.py` (NEW)

```python
def heuristic_action(agent_name, obs_flat, ammo, stamina, wrapper_state, game_config):
    """Scripted policy for G0/G1a/G1b. Returns int action ∈ [0,5]."""
```

Logic:
- **Archer:** if ammo==0 → NOOP. Else find nearest zombie, return ATTACK (4). If `archer_immobile`: never return movement.
- **Knight:** if stamina==0 → NOOP. Find nearest zombie. If within ~50px → ATTACK. Else pick movement toward zombie (approximate via x/y delta and agent orientation if available; for MVP always return ACTION=1 (forward) when zombie ahead).

**Obs parsing:** row 0 of `obs.reshape(27, 5)` = self; subsequent rows = other entities. Feature meaning from KAZ docs: `[x, y, type, type_extra, is_good]` (verify in pre-flight). Zombies identifiable by type marker.

### 4.3 `src/models/mappo_net.py` (MODIFIED)

**Add** `extra_dim: int = 0` to `__init__`.
- If `extra_dim > 0`: forward splits `obs` into `obs_entities` (first `entity_obs_dim` features) and `obs_extras` (last `extra_dim`).
- For `attention`/`attention_gru`: encoder processes `obs_entities` (reshaped to `(B, 27, 5)`) → pooled vector. Project `obs_extras` via `nn.Linear(extra_dim, hidden_dim)` and add to pool before backbone MLP.
- For `mlp`: concatenate and pass through as before.

**Backward compat:** `extra_dim=0` → identical V1/V2 behavior. No signature change to `get_action_and_value`. ~30 lines added.

### 4.4 `src/train_v3.py` (NEW)

Mirrors `src/train.py` with these differences:
- `--game` ∈ `{g2, g3, g4, g5}` (required)
- Auto-derive arch: `g2 → attention`; `g3/g4/g5 → attention_gru`
- `extra_dim=5` hardcoded (matches wrapper augmentation)
- `--ammo_mode` ∈ `{global, individual}` (default `individual`, should be passed from mega script)
- Uses `KAZWrapperV3`
- Checkpoints → `models/v3/{game}/final.pt`; TB → `results/tensorboard_v3/{game}/`
- New TB scalars: `metric/score`, `metric/failures`, `metric/archer_kills`, `metric/knight_kills`, `metric/archer_attacks`, `metric/knight_attacks`, `metric/ammo_pct_remaining`, `metric/stamina_pct_remaining`
- Transfer learning via `--transfer_from` (identical to V1 logic)
- Print-line banner including device, game, arch, ammo_mode, duration

### 4.5 `src/evaluate_v3.py` (NEW)

Supports **all 7 games**. `--game` ∈ `{g0, g1a, g1b, g2, g3, g4, g5}`.
- G0/G1a/G1b: use heuristic from `src/policies/heuristic.py`. No `--checkpoint` needed.
- G2–G5: load `--checkpoint`.
- `--seed 42` default. **Each episode** uses `seed = args.seed + episode_index` so that
  episode 0 of G2-eval has the same zombie pattern as episode 0 of G3-eval, etc.  This is
  done via explicit `env.reset(seed=args.seed + ep)` per episode (override the default
  `_seeded` behavior of the wrapper by always passing an explicit seed).
- Supports `--deterministic`, `--record`.
- Output JSON schema in §4.6.

### 4.6 Metrics JSON schema

`results/v3/{game}_eval_results[_det].json`:
```json
{
  "game": "g3", "episodes": 10, "seed": 42, "deterministic": false,
  "score_mean": 12.3, "score_std": 2.1,
  "stamina_pct_expended_mean": 0.82,
  "stamina_pct_expended_per_knight": [0.79, 0.85],
  "ammo_mode": "individual",                    // or "global"; absent for G0
  "ammo_pct_expended_mean": 0.96,                // global mode: single pool pct; individual: mean across archers
  "ammo_pct_expended_per_archer": [0.95, 0.97],  // only meaningful for individual; nulls for global
  "kills_archer_mean": 7.5, "kills_knight_mean": 4.8,
  "attacks_archer_mean": 14.2, "attacks_knight_mean": 23.4,
  "failures_mean": 1.2, "episode_length_mean": 443.0,
  "per_episode": [...]
}
```

### 4.7 `src/phase_artifacts_v3.py` (NEW)

Generates:
1. `results/v3/score_evolution.png` — bar chart of mean score G0 → G5
2. `results/v3/metric_breakdown.png` — 2×2 grid: %ammo, %stamina, kills (A vs K), attacks (A vs K)
3. `results/v3/saliency_v3.png` — attention heatmaps G3 vs G5 (does lock focus attention?)
4. `results/v3/demo_sidebyside_v3.mp4` — G0 heuristic vs G5 learned, same seed

### 4.8 `scripts/mega_train_v3.sh` (NEW, `chmod +x`)

```bash
#!/bin/bash
set -euo pipefail
mkdir -p models/v3 results/v3 results/tensorboard_v3
PY=./.venv/bin/python

echo "[1/7] G0 heuristic eval..."
$PY src/evaluate_v3.py --game g0 --episodes 10 --seed 42

echo "[2/7] G1a heuristic eval (global pool)..."
$PY src/evaluate_v3.py --game g1a --episodes 10 --seed 42

echo "[3/7] G1b heuristic eval (individual pool)..."
$PY src/evaluate_v3.py --game g1b --episodes 10 --seed 42

AMMO_MODE=$($PY -c "
import json
a = json.load(open('results/v3/g1a_eval_results.json'))
b = json.load(open('results/v3/g1b_eval_results.json'))
print('global' if a['score_mean'] >= b['score_mean'] else 'individual')
")
echo "  -> ammo_mode = $AMMO_MODE (used for G2-G5)"
echo "$AMMO_MODE" > results/v3/ammo_mode.txt

echo "[4/7] Training G2 (attention)..."
$PY src/train_v3.py --game g2 --ammo_mode "$AMMO_MODE" --total_timesteps 1000000

echo "[5/7] Training G3 (attn+GRU, <-G2)..."
$PY src/train_v3.py --game g3 --ammo_mode "$AMMO_MODE" --total_timesteps 1500000 \
    --transfer_from models/v3/g2/final.pt

echo "[6/7] Training G4 (+lock, <-G3)..."
$PY src/train_v3.py --game g4 --ammo_mode "$AMMO_MODE" --total_timesteps 1500000 \
    --transfer_from models/v3/g3/final.pt

echo "[7/7] Training G5 (+pragmatic, <-G4)..."
$PY src/train_v3.py --game g5 --ammo_mode "$AMMO_MODE" --total_timesteps 1500000 \
    --transfer_from models/v3/g4/final.pt

echo "V3 TRAINING COMPLETE. Tell Cascade to run evaluation pipeline."
```

---

## 5. Training (USER runs)

```bash
bash scripts/mega_train_v3.sh
```

**Watch in TensorBoard:**
- `metric/score` — should monotonically improve G2 → G5
- `metric/failures` — should decrease as teamwork improves
- `policy/entropy` — collapse below 0.2 early → add `--min_entropy 0.3` (feature already in train.py, carry to train_v3.py)
- Transfer: G3←G2 expect ~22/26 tensors (GRU new); G4←G3, G5←G4 expect 26/26

---

## 6. Evaluation Pipeline (CASCADE runs after training)

```bash
# Heuristic
for g in g0 g1a g1b; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --episodes 10 --seed 42
done

# Learned — stochastic + deterministic
for g in g2 g3 g4 g5; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3/$g/final.pt --episodes 10 --seed 42
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3/$g/final.pt --episodes 10 --seed 42 --deterministic
done

# Demos
./.venv/bin/python src/evaluate_v3.py --game g0 --episodes 3 --seed 42 --record
./.venv/bin/python src/evaluate_v3.py --game g5 --checkpoint models/v3/g5/final.pt --episodes 3 --seed 42 --record

# Artifacts
./.venv/bin/python src/phase_artifacts_v3.py
```

---

## 7. Documentation Updates (CASCADE)

- **`results/v3/ablation_table.md`** (NEW) — generate from 7 eval JSONs
- **`README.md`** — full rewrite: V3 narrative, 7-game table, game progression, quick start with mega script
- **`docs/report.tex`** + PDF — rewrite abstract/intro/methods/results/discussion; new figures; new ablation table; recompile with `pdflatex ... && bibtex ... && pdflatex x2`
- **`docs/presentation_guide.md`** — new slide-by-slide; include "Why we pivoted from V2" (1 slide, brief & honest)
- **`plan.md`** — add "V3: Structured Teamwork" roadmap section; mark V1/V2 as superseded
- **Keep for history:** `phase4_observations.md`, `models_v1_backup/`, `v2_death_penalty_plan.md`

---

## 8. Commit Strategy (branch: `v3-overhaul`)

**Commit 1** (after §4 implementation):
```
feat(v3): wrapper, heuristic policies, model extras, training & eval scripts
```

**Commit 2** (after training + evaluation):
```
feat(v3): training outputs and evaluation results for all 7 games
```

**Commit 3** (after docs):
```
docs(v3): README, report, presentation guide rewrite
```

Push `v3-overhaul`. Merge to main only after teammate approval.

---

## 9. Pre-flight Checks (BEFORE implementing §4)

Short smoke script that runs in <60s:

1. **Action mapping:** Send each action 0..5 to a knight for 10 steps. Confirm action 4 is attack, others cause movement/rotation. Record which.
2. **FPS:** `env.metadata['render_fps']` == 15 → `max_cycles = 450`.
3. **Zombie iteration:** `raw.zombie_list` yields sprites with `.rect.centerx/.centery`.
4. **Screen dims:** `raw.screen.get_size() == (1280, 720)`.
5. **Obs augmentation shape:** after §4.1.7 append, per-agent obs shape = `(140,)`.
6. **Transfer check:** G2-arch (attention, extra_dim=5) ↔ G3-arch (attention_gru, extra_dim=5) — should share 22+ tensors.
7. **KAZ constants:** import `ZOMBIE_Y_SPEED`, `KNIGHT_Y_SPEED` from KAZ `const` module; fall back to 0.5, 4.0 px/step if missing.

If any fails → update values in §2 before training.

---

## 10. Risk Register

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | KAZ internal attrs (`zombie_list`) may rename | Pin pettingzoo version; wrap accesses in try/except |
| 2 | Zombie killed AND crossed in same step (ambiguous) | If any positive reward this step → skip crossing count |
| 3 | Hard action masking → zero gradients for invalid actions | Start G4/G5 with **soft penalty** (−0.1 for invalid); promote to hard mask only if stable |
| 4 | V1 checkpoints incompatible (obs 135 vs 140) | Accepted — V3 trains G2 from scratch |
| 5 | Stamina pool too small/large | Monitor `stamina_pct_expended`; adjust pool to 50 or 150 before re-run |
| 6 | Failure signal too sparse | Kill reward dominates; failure is soft nudge — acceptable |
| 7 | Pragmatic override never fires in G5 | Log override count; if <1/ep, reduce lock radius or raise zombie speed |
| 8 | Heuristic too weak (G0 score ≈ 0) | Tune during pre-flight §9; iterate nearest-zombie targeting |
| 9 | Training time blow-up | Monitor SPS; reduce G4/G5 budgets to 1M if SPS < 300 |
| 10 | Obs parsing convention wrong (heuristic) | Verify `[x, y, type, type_extra, is_good]` via pre-flight; fix before eval |
| 11 | Patrol role oscillation — knight bounces in/out of end zone | If observed, tighten hard mask in G4 (cannot cross end-zone boundary) and accept soft oscillation in G3 |
| 12 | Archer-immobile heuristic for G1a/G1b produces zero movement signals so observation stays stale | KAZ still moves zombies each step; archer obs updates. Verified acceptable. |

---

## 11. Open Questions (TEAM DECISION before implementation)

These are ambiguous in the pseudoplan and I'm picking reasonable defaults. Flag to change any:

1. **G0 policy:** heuristic (my default) or trained unconstrained MLP? Pseudoplan doesn't specify.
2. **Stamina pool size:** 100 (my default) — may need tuning. Pseudoplan doesn't give a number.
3. **Global ammo pool size:** 30 (my default). Pseudoplan says individual = ceil(global/archers), but doesn't set global.
4. **Role bonus magnitude:** 0.001/step (my default). Can be 0 if you want zero shaping.
5. **Soft vs hard action mask for G4/G5:** default soft (penalty) for training stability, can escalate to hard.
6. **Pragmatic override (G5):** env-level deterministic logic (my interpretation) vs. policy-learned decision. Plan assumes env-level.
7. **Heuristic knight movement logic:** "always move forward if zombie ahead" is naive. Can tune but adds complexity.
8. **G2 reward shaping for sniper/hunter split:** pseudoplan §7c says "archers go for distant, knights for close" — do we (a) add a small per-kill bonus when archer kills distant / knight kills close, or (b) let the policy learn naturally from raw kills + failure signal? Default: **(b) natural**. Easier to tune out later.
9. **Lock cold-start:** On episode reset (cycle 0), no zombies exist → no locks possible. In G4/G5, archers cannot attack on step 0. Accepted; zombies spawn within a few steps.
10. **Parameter sharing across role-differentiated agents:** One policy network handles archer, forward knight, patrol knight, and end-zone-locked knight. The role/lock features in obs should suffice, but if learning is unstable, split into per-role policies (adds ~4× params).

---

## 12. What's Explicitly Dropped from V1/V2

- Fog (Gaussian noise on obs) — not in V3
- Death penalty — replaced by failure signal
- 60/40 team reward blend — replaced by shared failure signal
- Per-step stamina *cost* — replaced by stamina *pool*
- `--min_entropy` default off — available but off unless training issues arise

V1 and V2 artifacts remain in `results/` untouched. V3 writes to `results/v3/` to keep separation clean.
