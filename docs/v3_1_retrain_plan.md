# V3.1 Retraining Plan: Force Failures + Fix Knight Passivity

> **Branch:** `v3-overhaul` (continue; do NOT fork a new branch)
> **Source of truth:** `docs/v3_implementation_plan.md` (V3 design) + this addendum
> **Scope:** Parameter-only retraining. Zero architecture changes, zero code redesign.
> Goal: make the failure signal actually fire and eliminate knight passivity in G4/G5
> so the learned progression becomes a defensible monotonic story.

---

## 1. Why We Are Doing This

Brutal facts from V3 results (`results/v3/ablation_table.md`):

| Problem | Evidence | Consequence |
|---------|----------|-------------|
| Failure signal never fires | 0 / 70 episodes had any zombie cross | The V3 reward innovation (`−1.0` per crossing) is dead weight; models train on kill reward only |
| G2 < G1a heuristic | 2.40 vs 3.70 | "Added deep learning, performance dropped 35%" is undefendable |
| Knight passivity in G4/G5 | 0.0–0.5 attacks/ep vs 28.9 in G3 | 15% lock radius is too tight; forward knight rarely has a target |
| G1a = G1b exactly | 3.70 ± 1.19 identical to 4 sig figs | Ammo-mode ablation has no signal in sparse env |

Root cause for 3 of the 4: **spawn rate is too low** — the env is too easy for 4 agents,
so there are never enough zombies to force defense, exhaust the ammo pool meaningfully,
or saturate the knight's lock region.

Root cause for the 4th (knight passivity): the 15% lock radius was chosen before
we knew knights would see so few targets.

**Both root causes are single-parameter fixes.** No code redesign needed.

---

## 2. Parameter Changes (exact values, commit these)

### 2.1 Environment (`KAZWrapperV3` defaults)

| Parameter | V3 value | **V3.1 value** | Why |
|-----------|----------|----------------|-----|
| `spawn_rate` | 20 | **8** | ~56 zombies/episode (vs ~22 in V3). Forces failures when defense is imperfect. |
| `max_zombies` | 10 | **20** | Allows the denser spawn to actually accumulate on screen. |
| `knight_lock_radius_fraction` | 0.15 | **0.25** | 320 px radius; knights will have lockable targets consistently. |
| `global_ammo_pool` | 30 | **60** | Scale with 2.5× zombie count; keep expending ≈100% so discipline is meaningful. |
| `individual_ammo_pool` | 15 | **30** | `ceil(60/2)`. |
| `knight_stamina_pool` | 100 | **150** | Scale with expected attack count (avoid early depletion). |
| All other params (duration, agents, end zone, etc.) | unchanged | unchanged | |

**Why scale ammo/stamina proportionally?** If we raise spawn_rate without raising
pools, archers exhaust ammo by step ~120 and then sit helpless for 330 steps while
zombies stream in. That would inflate failures artificially (not because the policy
failed, but because the pool was undersized for the pressure). Scaling keeps the
"discipline" signal informative — agents still run out, just at a reasonable time.

### 2.2 Training budgets (unchanged)

| Game | Steps | Arch | Transfer |
|------|-------|------|----------|
| G2 | 1,000,000 | Attention | scratch |
| G3 | 1,500,000 | Attn + GRU | ← G2 |
| G4 | 1,500,000 | Attn + GRU | ← G3 |
| G5 | 2,000,000 | Attn + GRU | ← G4 |

Total ~6M steps, ~3.5 h on RTX 5070 Ti.

**Transfer is fine** even though env params change — the policy inputs (140-dim obs)
and outputs (6-way discrete action) are identical.

### 2.3 Evaluation (unchanged)

- 10 stochastic + 10 deterministic episodes per game
- Seed convention: `42 + ep_idx`, identical across games
- Same metrics JSON schema as V3

---

## 3. Mandatory Pre-Flight Check (DO THIS FIRST, ~2 min)

Before touching training, verify the new parameters actually produce failures. Do
this by reusing `src/evaluate_v3.py` after the wrapper default change (§4.1) is in place:

```bash
# Dry run: 3 episodes per game, check failures
for g in g0 g1a; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --episodes 3 --seed 42 \
      --results_dir /tmp/v3_1_preflight
done

# Extract failure counts
./.venv/bin/python -c "
import json
for g in ['g0','g1a']:
    d = json.load(open(f'/tmp/v3_1_preflight/{g}_eval_results.json'))
    print(f'{g}: failures_mean = {d[\"failures_mean\"]:.2f}')
"
```

(Use `src/evaluate_v3.py` rather than reconstructing the heuristic call
convention by hand — the actual signature of `heuristic_action` is whatever the V3
implementation settled on, and we don't want to second-guess it.)

**Acceptance criteria:**
- `g0` mean failures > 0 (heuristic should not be perfect under pressure)
- `g1a` mean failures >= 2 (immobile archers should struggle)

**If g1a mean failures < 2:** bump `spawn_rate` to 6, re-check. Keep halving until acceptance met.
**If g0 mean failures > 15:** too punishing; relax `spawn_rate` to 10.

Lock the spawn_rate that passes this check. Write it to `results/v3_1/spawn_rate_used.txt`.

---

## 4. Code Changes (minimal — 2 files)

### 4.1 `src/wrappers/kaz_wrapper_v3.py`

Update default values in `__init__`:

```python
def __init__(
    self,
    game_level: str,
    duration_seconds: int = 30,
    num_archers: int = 2, num_knights: int = 2,
    global_ammo_pool: int = 60,           # was 30
    individual_ammo_pool: int = 30,       # was 15
    knight_stamina_pool: int = 150,       # was 100
    end_zone_fraction: float = 0.20,
    knight_lock_radius_fraction: float = 0.25,   # was 0.15
    spawn_rate: int = 8,                  # was 20
    max_zombies: int = 20,                # was 10
    ...
):
```

**That's it.** No structural changes, no new logic.

### 4.2 `scripts/mega_train_v3.sh`

No changes required; the script already passes `--ammo_mode` from heuristic results.
The new env params are applied via wrapper defaults.

### 4.3 Output paths — use a new sub-directory to avoid clobbering V3

All retraining outputs go to:
- `models/v3_1/{g2,g3,g4,g5}/final.pt` (NOT `models/v3/`)
- `results/v3_1/` (NOT `results/v3/`)
- `results/tensorboard_v3_1/` (NOT `results/tensorboard_v3/`)

**Code changes required (these ARE new CLI args, flag them as such when implementing):**

1. **`src/train_v3.py`** — add `--output_suffix` arg (default `""`). When set, appended
   to output paths: `models/v3{suffix}/{game}/`, `results/tensorboard_v3{suffix}/{game}/`.
   Use `--output_suffix _1` for V3.1 runs.
2. **`src/evaluate_v3.py`** — add `--results_dir` arg (default `results/v3`). All JSON
   and demo outputs go here.
3. **`src/phase_artifacts_v3.py`** — add `--results_dir` (where to READ eval JSONs) and
   `--output_dir` (where to WRITE figures). Both default to `results/v3`.

These are first-class code changes, not runtime config. Include them in Commit A if
possible; otherwise Commit A is plan-only and these go in Commit B's first step.

**Why preserve V3 results?** So the report can show V3 vs V3.1 as a controlled
parameter sweep — a real experiment rather than "we overwrote everything."

### 4.4 `scripts/mega_train_v3_1.sh` (NEW)

Copy of `mega_train_v3.sh` with:
- Eval commands use `--results_dir results/v3_1`
- Train commands use `--output_suffix _1`
- Transfer-from paths use `models/v3_1/...`
- Final banner says "V3.1 TRAINING COMPLETE"

---

## 5. Training (USER runs one command)

```bash
bash scripts/mega_train_v3_1.sh
```

**Watch in TensorBoard during training:**
- `metric/failures` — should now be > 0 in rollouts
- `metric/knight_attacks` — should be > 5/ep in G4, not 0
- `policy/entropy` — if collapses below 0.2 before 200K steps, abort and retry with
  `--min_entropy 0.3` (already supported in train.py)

---

## 6. Evaluation (CASCADE runs after training)

```bash
# Heuristic (new params produce new numbers)
for g in g0 g1a g1b; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --episodes 10 --seed 42 \
      --results_dir results/v3_1
done

# Learned (stoch + det)
for g in g2 g3 g4 g5; do
  ./.venv/bin/python src/evaluate_v3.py --game $g \
      --checkpoint models/v3_1/$g/final.pt --episodes 10 --seed 42 \
      --results_dir results/v3_1
  ./.venv/bin/python src/evaluate_v3.py --game $g \
      --checkpoint models/v3_1/$g/final.pt --episodes 10 --seed 42 \
      --deterministic --results_dir results/v3_1
done

# Demos (G0 heuristic vs G5 learned, same seed)
./.venv/bin/python src/evaluate_v3.py --game g0 --episodes 3 --seed 42 --record \
    --results_dir results/v3_1
./.venv/bin/python src/evaluate_v3.py --game g5 \
    --checkpoint models/v3_1/g5/final.pt --episodes 3 --seed 42 --record \
    --results_dir results/v3_1

# Artifacts
./.venv/bin/python src/phase_artifacts_v3.py \
    --results_dir results/v3_1 --output_dir results/v3_1
```

---

## 7. Documentation Updates (CASCADE)

### 7.1 `results/v3_1/ablation_table.md` (NEW, auto-generated)
Same schema as V3; overwrites nothing.

### 7.2 `README.md`
- Replace headline results table with V3.1 numbers
- Add a "V3 → V3.1 parameter sweep" subsection explaining the spawn rate / lock radius change and the reason (failures must actually fire)
- Update "Status" table: V3.1 is current

### 7.3 `docs/report.tex`
- Replace Table~\ref{tab:main} with V3.1 numbers
- Replace Table~\ref{tab:kills} with V3.1 numbers
- Update paragraph on "Zero failures throughout" → new paragraph showing what the
  failure signal actually does under pressure
- Update "Knight passivity" subsection to reflect new (hopefully reduced) passivity
- Update Figure references to V3.1 figures
- Keep old V3 numbers as an appendix table titled "Parameter Sensitivity: V3 vs V3.1".
  **Honesty caveat for the appendix:** V3 and V3.1 use identical eval seeds, but
  because `spawn_rate` differs, the actual zombie trajectories differ between the two.
  The comparison is therefore a parameter-sensitivity sweep under identical policies,
  NOT an apples-to-apples same-zombies benchmark. State this explicitly in the caption.
- Recompile PDF

### 7.4 `docs/presentation_guide.md`
- Replace Slide 5 numbers with V3.1
- Update "Why we pivoted from V2" slide to brief "V3 → V3.1 parameter fix" slide
  (honest: "we noticed failures weren't firing, bumped spawn rate, retrained")
- Update Slide 11 (limitations) — drop "zero failures in every game" item; add
  whatever new limitations surface

### 7.5 `plan.md`
- Add a `### Phase V3.1 — Pressure retraining ✅` sub-section under "V3 — Structured Teamwork"
- Log exact parameter changes + new results

---

## 8. Commit Strategy (on `v3-overhaul`)

Three commits, identical pattern to V3:

**Commit A** (plan only — this file):
```
docs(v3.1): retraining plan for pressure + lock radius fix

Spawn rate 20 -> 8, max_zombies 10 -> 20, lock radius 15% -> 25%,
ammo pool 30 -> 60, stamina pool 100 -> 150.

Motivation: V3 failure signal never fired (0/70 eps); G4/G5 knights
had 0 attacks/ep due to tight lock radius. V3.1 addresses both with
parameter-only changes (no architectural redesign).

Outputs to models/v3_1/ and results/v3_1/ (V3 preserved).
```

**Commit B** (after training + eval):
```
feat(v3.1): retrained G2-G5 under pressure + full evaluation

[summary of new numbers, especially failures_mean per game]
```

**Commit C** (docs):
```
docs(v3.1): README/report/presentation guide updated with V3.1 results
```

Push to `v3-overhaul`. Keep the `v3-overhaul` branch — V3.1 is a refinement of V3,
not a new experiment branch.

---

## 9. Risk Register & Mitigations

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | Pre-flight fails (g1a failures still = 0 at spawn_rate=8) | Halve spawn_rate until pass. If still fails at rate=2, raise `max_zombies` to 30. |
| 2 | Training diverges — entropy collapses | Use `--min_entropy 0.3`. Already available in train.py; inherit into train_v3.py. |
| 3 | G2 STILL < heuristic G1a | If this happens, the problem is not spawn rate — it's the learning objective. Fallback: run an unconstrained G2 variant (drop ammo_on) to verify the learning pipeline works at all. |
| 4 | Knight passivity persists at radius=0.25 | Read TB `metric/knight_attacks`. If <5/ep by 1M steps in G4, widen to 0.35 and retrain JUST G4/G5 (skip G2/G3 re-training). Document the sweep. |
| 5 | Failures dominate reward signal (agents freeze) | Check TB `metric/failures`. If >15/ep by 500K steps, the failure penalty is too harsh; lower `p_f` from 1.0 to 0.5 in the wrapper. |
| 6 | Transfer from V3 checkpoints fails | Do NOT transfer from V3 — V3.1 trains from scratch. (The plan as written does this: G2 trains from scratch, G3 transfers from V3.1-G2, etc.) |
| 7 | Ammo pool runs out too fast (archer does nothing after step 100) | Check TB `metric/ammo_pct_remaining`. If hits 0 before step 200 consistently, raise pool to 90/45. |
| 8 | Wall-clock exceeds overnight window | Reduce G4/G5 budgets to 1M each; still leaves G2/G3 full budgets. |
| 9 | New demo video doesn't illustrate failures | If after retraining no failures visible in recorded demo, record an extra demo at spawn_rate=4 for illustration only (not for eval table). |
| 10 | Report's "V3 vs V3.1 parameter sensitivity" table reveals V3 was fundamentally flawed | Good. That's science. Frame as "we identified a calibration issue, fixed it, report both for transparency." |

---

## 10. Fallback Tree (if primary plan fails)

```
Did pre-flight produce failures > 2 in g1a?
├── YES → proceed with spawn_rate=8 → continue to training
└── NO
    ├── Drop spawn_rate to 6 → re-check
    │   ├── YES → proceed
    │   └── NO → drop to 4 → re-check
    │       ├── YES → proceed
    │       └── NO → raise max_zombies to 30, spawn_rate back to 6 → re-check
    │           ├── YES → proceed
    │           └── NO → abort; escalate to user (env too resilient)

After training, did any game get knight_attacks > 5 in G4/G5?
├── YES → proceed to docs
└── NO → retrain ONLY G4/G5 with lock_radius=0.35 (save time; ~100min total)

Did G2 beat heuristic G1a?
├── YES → great, defensible progression
└── NO
    ├── Check TB entropy — if collapsed, retrain G2 with --min_entropy 0.3
    └── If entropy fine, accept: report as "learned G2 matches heuristic ±noise;
        the learning gain appears at G3 via roles." (Still defensible given +96% G2→G3.)
```

---

## 11. Expected Outcomes (predictions, to sanity-check against after training)

These are the numbers I expect to see; use them as gut-check during training:

| Game | Expected score (stoch) | Expected failures | Expected knight attacks |
|------|-----------------------|-------------------|-------------------------|
| G0 (heuristic, unrestricted) | 18–25 | 2–5 | 25+ |
| G1a (heuristic, constrained) | 6–10 | 8–15 | 5–8 |
| G1b (heuristic, constrained) | 6–10 | 8–15 | 5–8 |
| G2 (learned attention) | 5–10 | 6–12 | 15–30 |
| G3 (+ roles + GRU) | **12–18** | 3–8 | 20–35 |
| G4 (+ locks) | 10–16 | 4–9 | 10–20 |
| G5 (+ pragmatic) | 11–17 | 3–7 | 10–20 |

**If G2 < G1a after retraining, stop and debug before running G3–G5** — it means the
learning objective has a real issue, not a calibration one.

---

## 12. Explicit Non-Goals

- **Do NOT redesign the reward function.** The failure signal + kill reward structure is fine; it just needs to actually fire.
- **Do NOT add new architectures.** Same MAPPO + Attn / Attn+GRU.
- **Do NOT change the ablation ladder.** Same 7 games.
- **Do NOT delete V3 artifacts.** Keep `models/v3/`, `results/v3/` intact for the parameter-sensitivity appendix.
- **Do NOT re-run pre-flight mid-training.** Pre-flight once, lock the params, train.

---

## 13. Success Criteria (how we know V3.1 fixed V3)

Ship V3.1 as the headline result if ALL of these are true:
1. ✅ Mean failures > 0 in G1a/G1b heuristic eval
2. ✅ At least one learned game (G3 or higher) shows failures > 0 in stochastic eval — proving agents face real pressure
3. ✅ G3 ≥ G2 (progression is monotonic through G3)
4. ✅ G4/G5 knight attacks > 5/ep (passivity resolved)
5. ✅ Best learned game ≥ best heuristic-constrained game (G1a/G1b)

If 3/5 pass: ship V3.1 with honest framing about what's still not fixed.
If ≤2/5 pass: the parameter sweep wasn't enough; escalate to user for re-scoping before presentation.
