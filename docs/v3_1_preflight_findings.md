# V3.1 Pre-Flight Findings — Plan ABORTED

> **Status:** V3.1 retraining plan (`docs/v3_1_retrain_plan.md`) **aborted** before any code
> or training changes were committed. Root cause of V3's zero-failure pathology is a
> code defect in the failure detector, not a parameter calibration issue. The plan's
> "parameter-only, no code redesign" premise is invalidated; re-scope required.

## 1. What was attempted

Per the plan's strict ordering, the §3 pre-flight check was executed after staging the
§4 wrapper default changes in a working tree (not committed). Staged changes matched
§2.1 exactly:

| Parameter | V3 default | V3.1 staged |
|-----------|------------|-------------|
| `spawn_rate` | 20 | 8 |
| `max_zombies` | 10 | 20 |
| `knight_lock_radius_fraction` | 0.15 | 0.25 |
| `global_ammo_pool` | 30 | 60 |
| `individual_ammo_pool` | 15 | 30 |
| `knight_stamina_pool` | 100 | 150 |

Acceptance gate from §3: `g1a` heuristic must yield `failures_mean ≥ 2`
(and `g0 failures_mean > 0`) over 3 episodes with seed 42.

## 2. Pre-flight results — every fallback node failed

The §10 fallback tree was walked exhaustively. All configurations produced
`failures_mean = 0.00` for both `g0` and `g1a`:

| `spawn_rate` | `max_zombies` | g0 failures | g1a failures | g0 score | g1a score |
|---|---|---|---|---|---|
| 8 | 20 (plan target) | 0.00 | 0.00 | 18.00 | 5.00 |
| 6 | 20 | 0.00 | 0.00 | 14.67 | 9.00 |
| 4 | 20 | 0.00 | 0.00 | 15.00 | 9.67 |
| 6 | 30 (tree terminal) | 0.00 | 0.00 | 14.67 | 9.00 |
| 2 | 30 (diagnostic, past tree) | 0.00 | 0.00 | 30.00 | 14.33 |

This matches the fallback tree's abort node:

> `NO → abort; escalate to user (env too resilient)`

Note that spawn pressure demonstrably increased — g0 score jumped from 18.0 at
`spawn_rate=8` to 30.0 at `spawn_rate=2`, confirming more zombies were spawning,
reaching agents, and being killed. **Yet failures remained zero across every
configuration.**

## 3. Root cause — failure detector bug in `kaz_wrapper_v3.py`

Two combined issues in the current failure-event detector make the event effectively
unreachable regardless of spawn pressure:

```@/home/owner/ADLProject2/src/wrappers/kaz_wrapper_v3.py:471-484
        new_zombies = self._zombies()
        current_ids = {zid for zid, _, _, _ in new_zombies}
        any_kill = any(r > 0.5 for r in raw_rewards.values())
        failures_this_step = 0
        for zid, (zx, zy) in self.zombie_prev.items():
            if zid not in current_ids:
                # Disappeared — killed or crossed
                crossed = zy >= (self.SCREEN_H - self.ZOMBIE_Y_SPEED - 5)
                if crossed and not any_kill:
                    failures_this_step += 1
                elif crossed and any_kill:
                    # Ambiguous; per Risk #2 prefer the kill (skip cross)
                    pass
        self.failure_count += failures_this_step
        self.last_failures_this_step = failures_this_step
```

### 3.1 Global `any_kill` gate swallows concurrent crossings

`any_kill` is computed over **all agents** on the step. With 4 agents and 200–300
archer attacks per episode, a kill happens in a large fraction of time-steps.
Whenever a zombie actually crosses in the same step that any other zombie is killed,
the crossing is discarded. This is not a rare ambiguity — under the new spawn
pressure it is effectively the common case.

### 3.2 Agents intercept zombies mid-screen, so `crossed=False`

The `crossed` test requires `zy ≥ SCREEN_H − ZOMBIE_Y_SPEED − 5` — roughly the
bottom 15 px of the 720 px playfield. Observed behaviour:

- **G1a (immobile archers at the bottom):** a diagnostic `g1a` heuristic episode
  terminated at step **145/450** because all agents had been killed by incoming
  zombies. Those zombies disappeared from the roster not at `SCREEN_H − 15`
  but at the collision y of the archers they killed — so `crossed=False`, no failure.
- **G0 (free movement):** knights and archers range across the field and kill zombies
  far above the end-zone strip; the 15 px band is almost never reached.

Both conditions make the detector pathologically rare, so `failures_mean = 0.00`
is the equilibrium behaviour of the V3 wrapper — **not** a symptom of V3 being
"too easy."

## 4. Why the plan's premise is invalidated

The plan's §1 "Why We Are Doing This" attributes the zero-failure symptom to
under-pressured spawns and sets "parameter-only retraining" (§12) as a non-goal.
The pre-flight falsifies both:

1. Spawn pressure alone cannot drive `failures_mean > 0` at any level tested down
   to `spawn_rate=2`, `max_zombies=30`.
2. Fixing the symptom requires code edits inside the reward / event pipeline,
   which §12 explicitly forbids.

Continuing with V3.1 as written would reproduce the V3 zero-failure artifact with
different spawn parameters — wasting ~3.5 h of GPU time without addressing the
defect.

## 5. Disposition

- **No code changes committed.** The §4 edits (wrapper defaults,
  `--output_suffix` on `train_v3.py`, `--results_dir` on `evaluate_v3.py`, `--results_dir`/`--output_dir`
  on `phase_artifacts_v3.py`, `scripts/mega_train_v3_1.sh`) were staged locally
  and reverted after the pre-flight failed.
- **V3 artifacts preserved.** `models/v3/`, `results/v3/`, and
  `results/tensorboard_v3/` are untouched.
- **Working tree is clean against `v3-overhaul`.** Only this findings document
  and `docs/v3_1_retrain_plan.md` (committed separately on this branch) remain as
  additions.

## 6. Recommended re-scope options (for the next plan revision)

Ordered from smallest to largest scope:

1. **Minimal detector patch (one-line).** Remove the `any_kill` gate. Count every
   zombie that disappears within the bottom `ZOMBIE_Y_SPEED + 5` px as a failure.
   Tradeoff: tiny chance of miscounting when a zombie is killed *at* the end line.
2. **Detector patch + widened band.** (1) plus widen `crossed` to
   `zy >= 0.85 * SCREEN_H` so g1a-style interceptions at the archer line still
   register. Best coverage, biggest deviation from "parameter-only."
3. **Re-define failure as "agent died to zombie contact."** Agent death already
   fires natively in KAZ; failures_mean would become a non-trivial positive number
   immediately. This abandons the original "zombie reached the base" semantics.
4. **Drop the failure term entirely.** Report score = kills; remove the
   `failure_penalty` pathway from the reward composition. Clean but loses the
   V3 design intent of defensive pressure.

Any of (1)–(4) would require a follow-up plan document, new commit(s), and a
re-run of the §3 pre-flight acceptance gate before training.

## 7. Next action for the user

Pick a re-scope option from §6 (or propose another), update
`docs/v3_1_retrain_plan.md` accordingly, and re-run `docs/v3_1_retrain_plan.md §3`
pre-flight against the new detector. V3.1 training should remain deferred until
the pre-flight passes with `g1a failures_mean ≥ 2`.
