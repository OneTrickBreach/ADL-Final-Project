# V3.1 RETRAINING PLAN (FIXED) — Detector Rewrite + Parameter Retune

> **Branch:** `v3-overhaul` (continue)
> **Supersedes:** `docs/v3_1_retrain_plan.md` (that plan's "parameter-only" premise was
> falsified by `docs/v3_1_preflight_findings.md`).
> **Source of truth for what V3.1 tries to achieve:** the abort doc's §6 re-scope
> options; this plan adopts a hybrid of option 1 ("detector patch") and option 3
> ("agent deaths as failures") plus an optional episode-continuation override.
> **Scope:** minimal code fix to `kaz_wrapper_v3.py` (one method) + the V3.1 parameter
> changes + one-shot retrain.

---

## 1. What the aborted pre-flight actually proved

From `docs/v3_1_preflight_findings.md` + empirical verification today:

**The V3 failure detector is backwards.** It counts zombies that *disappear* from
`zombie_list`, then gates on `any_kill`. But in KAZ:

| Zombie event | Effect on `zombie_list` | Correct label |
|--------------|--------------------------|---------------|
| Arrow / sword kills zombie | **removed** from list (`arrow_hit`, `sword_hit` lines 480–491) | NOT a failure |
| Zombie crosses end line | **stays** in list; KAZ sets `raw.run = False` → episode ends | IS a failure |
| Zombie contacts archer/knight | **stays** in list; agent removed from `archer_list`/`knight_list`; zombie keeps moving | IS a failure |

So the V3 detector was computing "zombies killed by the team, that happened to be
low on screen when no other kill occurred that step" — effectively `min(rare
events) ≈ 0`, regardless of spawn pressure.

**Empirically verified today** (raw env, seed 42, `spawn_rate=8`):
- Passive agents (all NOOPs): all 4 agents dead by step 120; 0 zombies crossed;
  `raw.run=False` via `zombie_all_players`.
- Random actions: 1 zombie crossed at step 150; `raw.run=False` via
  `zombie_endscreen`; episode ended.

Both scenarios produce a rich failure signal if (and only if) the detector looks
at the right things.

---

## 2. The fix — rewrite the detector in `kaz_wrapper_v3.py`

**One method change.** No architectural changes, no reward-shape changes, no new
CLI arguments beyond the ones V3.1 already needed (`--output_suffix`,
`--results_dir`, `--output_dir`).

### 2.1 New detector semantics

A **failure** is logged whenever either of these occurs on a step:

1. **Line crossing:** a zombie in the *current* `zombie_list` has
   `centery >= SCREEN_H − ZOMBIE_Y_SPEED` and its `id()` has not been counted yet.
2. **Agent breached:** the size of `archer_list + knight_list` decreased this step
   (KAZ only shrinks those via zombie contact).

These two are additive. A single step may log multiple failures.

### 2.2 Required code changes in `src/wrappers/kaz_wrapper_v3.py`

**a) In `reset()` — add new tracking state:**
```python
self._crossed_ids = set()
raw = self._env.unwrapped
self._prev_alive_count = len(raw.archer_list) + len(raw.knight_list)
```

**b) Replace the current `any_kill`-gated disappearance scan with the code below.**
Locate it by searching for `any_kill = any(r > 0.5 for r in raw_rewards.values())` in
`kaz_wrapper_v3.py`; line numbers may have shifted from the findings doc's cite.
```python
# --- V3.1 failure detector: line-cross + agent-breach (ids + list shrinkage) ---
raw = self._env.unwrapped
CROSS_Y = self.SCREEN_H - self.ZOMBIE_Y_SPEED

failures_this_step = 0

# (a) zombies past the end line (still in list; KAZ does not remove crossers)
for z in raw.zombie_list:
    zid = id(z)
    if z.rect.centery >= CROSS_Y and zid not in self._crossed_ids:
        self._crossed_ids.add(zid)
        failures_this_step += 1

# (b) agents killed by zombie contact this step (archer_list / knight_list shrink)
cur_alive = len(raw.archer_list) + len(raw.knight_list)
if cur_alive < self._prev_alive_count:
    failures_this_step += (self._prev_alive_count - cur_alive)
self._prev_alive_count = cur_alive

self.failure_count += failures_this_step
self.last_failures_this_step = failures_this_step
```

**c) Remove the `zombie_prev` snapshot and `any_kill` computation** from the step
method — they are no longer used by the new detector. Delete the dict if
nothing else in the wrapper consumes it.

### 2.3 Optional (tier-2) episode-continuation override

Without this, a line cross ends the episode (because KAZ sets `raw.run=False`), so
the failure-per-episode count tops out at `1 + agent_deaths`. With it, training
rollouts can accumulate multiple crosses per episode → richer signal and a more
interesting "defense under pressure" learning target.

Insert immediately after the detector block:
```python
# Allow episode to continue past individual line crosses so the policy can
# learn from multiple failures per rollout. Only override when agents are
# still alive and we have not hit the natural truncation.
if (
    not raw.run
    and cur_alive > 0
    and self.cycle + 1 < self.max_cycles
):
    raw.run = True
    terms = {a: False for a in terms}

    # Prevent the just-counted crossers from re-triggering zombie_endscreen
    # on the very next step: either remove them, or push them out of bounds.
    for z in list(raw.zombie_list):
        if id(z) in self._crossed_ids and z.rect.centery >= CROSS_Y:
            raw.zombie_list.remove(z)
```

**Tier-2 risks:**
- KAZ internals (`self.frames`, `kill_list`) are not touched — this is
  non-invasive.
- Removing crossed zombies prevents double-counting and frees the `max_zombies`
  slot.
- If any future KAZ update also removes crossed zombies, `_crossed_ids` guards
  the double-count on our side.

**Decision:** Implement tier-2 unless pre-flight shows tier-1 alone already
satisfies the acceptance gate by a comfortable margin (`g1a failures_mean ≥ 4`).

### 2.4 What does NOT change

- Reward composition (`−1.0 × failures_this_step` on all living agents). Now it
  will actually fire.
- Observation augmentation (still 140-d).
- Action masking policy (soft, with optional hard for patrol-knight
  `lock_on`).
- Every other wrapper responsibility (ammo pools, stamina pools, target locks,
  pragmatic override, role assignment).

---

## 3. Parameter changes — reaffirmed from V3.1 original §2

Keep all six parameter changes from the aborted plan:

| Parameter | V3 | **V3.1** | Purpose |
|-----------|----|---------:|---------|
| `spawn_rate` | 20 | **8** | ~56 zombies/ep vs ~22 |
| `max_zombies` | 10 | **20** | Allow concurrent accumulation |
| `knight_lock_radius_fraction` | 0.15 | **0.25** | Knights actually have lockable targets |
| `global_ammo_pool` | 30 | **60** | Scale with zombie count |
| `individual_ammo_pool` | 15 | **30** | `ceil(60/2)` |
| `knight_stamina_pool` | 100 | **150** | Scale with expected attacks |

Rationale unchanged; see `docs/v3_1_retrain_plan.md §2` for the full discussion.

---

## 4. Revised pre-flight (MUST PASS before training)

Same structure as aborted plan's §3 but with the fixed detector in place.

```bash
# After 2.1+2.2 (and 2.3 if adopting tier-2) are applied to kaz_wrapper_v3.py:
for g in g0 g1a g1b; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --episodes 3 --seed 42 \
      --results_dir /tmp/v3_1_preflight
done

./.venv/bin/python -c "
import json
for g in ['g0','g1a','g1b']:
    d = json.load(open(f'/tmp/v3_1_preflight/{g}_eval_results.json'))
    print(f'{g}: failures_mean = {d[\"failures_mean\"]:.2f}, score_mean = {d[\"score_mean\"]:.2f}, ep_len_mean = {d[\"episode_length_mean\"]:.0f}')
"
```

### Acceptance gates

| Gate | Condition | Rationale |
|------|-----------|-----------|
| A | `g1a failures_mean ≥ 2` | The primary defect the aborted pre-flight exposed. |
| B | `g0 failures_mean > 0` | Unrestricted heuristic should occasionally fail under the new spawn pressure. |
| C | Detector does not inflate: `g0 failures_mean ≤ 2 × g0_kills_mean` | Sanity check against runaway false-positives. |
| D (if tier-2) | At least one of the three g1a episodes shows `ep_len > 150` | Proves the episode-continuation override is working (without it, g1a would always end by step ~150 when all agents die). |

### Fallback tree (much shorter than v1 because the fix is structural, not calibrational)

```
A passes?
├── YES → go to C.
│         C passes?
│         ├── YES → ship; proceed to training.
│         └── NO (detector over-counts) → tighten: raise CROSS_Y threshold from
│             SCREEN_H - ZOMBIE_Y_SPEED to SCREEN_H - 1, and/or drop the
│             agent-breach term. Re-run preflight.
└── NO
    └── A failing means agents do NOT die and zombies do NOT cross at
        spawn_rate=8 with the fixed detector. This would contradict today's
        empirical verification. Treat as anomaly: print the per-episode
        failure dict, attach to the findings doc, and escalate.
```

---

## 5. Code-change surface (full inventory)

| File | Change | Lines |
|------|--------|-------|
| `src/wrappers/kaz_wrapper_v3.py` | Replace detector (§2.2); add reset init (§2.1); (optional) add override (§2.3). Update 6 constructor defaults per §3. | ~30 |
| `src/train_v3.py` | Add `--output_suffix` CLI arg, append to model/TB dirs. | ~5 |
| `src/evaluate_v3.py` | Add `--results_dir` CLI arg, use for JSON + demo outputs. | ~5 |
| `src/phase_artifacts_v3.py` | Add `--results_dir` + `--output_dir`. | ~5 |
| `scripts/mega_train_v3_1.sh` (NEW) | Same as V3 script but with `--output_suffix _1`, `--results_dir results/v3_1`, transfer from `models/v3_1/...`. | ~50 |

All V3 artifacts (`models/v3/`, `results/v3/`, `results/tensorboard_v3/`) remain
untouched so that the final report can include a V3 vs V3.1 parameter-sensitivity
appendix.

---

## 6. Training & evaluation (no change from V3.1 original)

See `docs/v3_1_retrain_plan.md §5–§7` for the full commands. Only the pre-flight
step is updated here; everything downstream is identical.

One-shot:
```bash
bash scripts/mega_train_v3_1.sh
```

Expected wall-clock on RTX 5070 Ti: ~3.5 h (6M timesteps total).

---

## 7. Expected outcomes after the detector fix

Updated prediction table (note the `failures` column is now meaningful):

| Game | Score (stoch) | Failures/ep | Knight attacks/ep |
|------|---------------|-------------|-------------------|
| G0 (heuristic, free) | 18–25 | 1–4 | 25+ |
| G1a (heuristic, immobile archers) | 6–10 | **4–6** (all agents die under pressure) | 5–8 |
| G1b (heuristic, immobile archers) | 6–10 | **4–6** | 5–8 |
| G2 (learned, attention) | 5–10 | 3–6 | 15–30 |
| G3 (+roles +GRU) | **12–18** | **1–4** | 20–35 |
| G4 (+locks) | 10–16 | 2–5 | 10–20 |
| G5 (+pragmatic) | 11–17 | 1–4 | 10–20 |

**Ship-gate criteria** (identical to V3.1 original §13) must all hold after eval:
1. g1a/g1b failures_mean > 0 ✓ (trivially true with fixed detector)
2. At least one learned game shows failures > 0 stochastically
3. G3 ≥ G2
4. G4/G5 knight attacks > 5/ep
5. Best learned ≥ best constrained heuristic

---

## 8. Commit strategy

**Commit 0 — this plan (happens NOW, before implementation):**
```
docs(v3.1): fixed plan — detector rewrite supersedes parameter-only approach
```

Then after implementation, three commits as in V3:

**Commit A — detector fix + V3.1 scaffolding (implementation):**
```
feat(v3.1): rewrite failure detector + V3.1 parameter retune + CLI plumbing

Root-cause fix for the zero-failures bug exposed by the pre-flight of
docs/v3_1_retrain_plan.md:
  * src/wrappers/kaz_wrapper_v3.py: detector rewrite — line-cross on
    current zombie_list + agent-breach via archer/knight list shrinkage.
    Optional episode-continuation override to accumulate multiple
    failures per rollout.
  * src/wrappers/kaz_wrapper_v3.py: constructor defaults retuned for
    V3.1 (spawn_rate 20->8, max_zombies 10->20, lock_radius 0.15->0.25,
    ammo 30/15 -> 60/30, stamina 100 -> 150).
  * src/train_v3.py: --output_suffix CLI arg.
  * src/evaluate_v3.py: --results_dir CLI arg.
  * src/phase_artifacts_v3.py: --results_dir and --output_dir CLI args.
  * scripts/mega_train_v3_1.sh: orchestration targeting models/v3_1/ and
    results/v3_1/ so V3 artifacts are preserved for the sensitivity
    appendix.

Pre-flight now passes (g1a failures_mean >= 2, g0 failures_mean > 0)
under the new detector. Training deferred to the user's one-shot
invocation.
```

**Commit B:** retrained models + eval results (identical shape to V3 commit B).
**Commit C:** README / report / presentation / plan.md updates.

---

## 9. Risk register (delta from V3.1 original)

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | Detector over-counts agent deaths (e.g. if an agent leaves the list for any non-zombie reason) | Verified: KAZ only removes agents via `kill_list` which is populated exclusively from `zombie_hit_archer` / `zombit_hit_knight`. Safe. |
| 2 | Episode-continuation override confuses KAZ internal state | Override only flips `run` back and removes already-counted crossers; touches no other KAZ state. Smoke-test in pre-flight via gate D. |
| 3 | Multiple crosses per rollout over-drive the penalty term, destabilising training | **Preemptive cap:** in reward computation, apply `penalty = -1.0 * min(failures_this_step, 3)` to all living agents. Keep the raw counter (for metrics) unclamped. Watch TB `metric/failures`; if >10/ep consistently, consider dropping tier-2. |
| 4 | Tier-2 override lets zombies "teleport" off-screen and agents don't see them | By removing crossed zombies from `zombie_list` after counting, they simply disappear — same effect as if KAZ had removed them. |
| 5 | Acceptance gate D (tier-2 only) triggers false pass if an agent dies just as a zombie crosses | Rare edge case; acceptable. |

---

## 10. Explicit non-goals

- No new architecture. Still MAPPO + Attn / Attn+GRU.
- No reward re-weighting. Same `−1.0` per failure (clamped per-step per Risk #3 to keep training stable; the coefficient itself is unchanged).
- No obs-dim change. Still 140.
- No modification to games G2's arch or to the ablation ladder.
- No attempt to retrain V3 models on the fixed detector (would invalidate V3
  results for nothing).

---

## 11. One-line summary

**V3 counted the wrong thing; V3.1 counts the right thing, under heavier pressure.**
