## ADL Final Project — Roadmap

Live ledger for all iterations of this study. Most recent work is **V3.1**
("Structured Teamwork Under Pressure"). V3 is preserved for the sensitivity
appendix; V1 and V2 remain unchanged on their own branches.

---

## V3.1 — Detector Rewrite + Pressure Retune (current)

> Branch: `v3-overhaul` (continued) · Reference spec: `docs/v3_1_retrain_plan_fixed.md`
> Supersedes V3 which had a backwards failure detector (see `docs/v3_1_preflight_findings.md`).

**Thesis.** The V3 thesis (structure > reward magnitude) was right, but V3's
numbers were measured under the wrong instrument: the failure detector
double-counted kills as crosses and the default pressure was low enough that
`failures_mean = 0` across all 7 games. V3.1 fixes both: a rewritten detector
(line-cross on `centery` + agent-breach via sprite-list delta) and retuned
pressure (spawn_rate 20→8, max_zombies 10→20, lock radius 0.15→0.25 W, ammo
30/15→60/30, knight stamina 100→150).

### Phase V3.1-1 — Pre-flight ✅

Pre-flight gates all PASS (commit `0796b60`):
* **A**: `g1a failures_mean ≥ 2` (3.33 observed)
* **B**: `g0 failures_mean > 0` (3.00 observed)
* **C**: `g0 failures_mean ≤ 2 × g0_kills_mean` (3.00 ≤ 36.00)
* **D**: ≥1 g1a episode with `ep_len > 150` (all 3 test eps: 151 / 175 / 173)

### Phase V3.1-2 — Implementation ✅ (commit `0796b60`)

* `src/wrappers/kaz_wrapper_v3.py` — detector rewrite (line-cross + agent-breach),
  tier-2 episode-continuation override, Risk #3 reward cap (`min(failures, 3)`),
  six constructor defaults retuned per §3 of the fixed plan.
* `src/train_v3.py` — `--output_suffix` CLI arg routes checkpoints to
  `models/v3_1/` and TB logs to `results/tensorboard_v3_1/`.
* `src/evaluate_v3.py` — `--results_dir` CLI arg for JSON + demo outputs,
  `ammo_mode.txt` fallback first checks `--results_dir`, then `results/v3/`.
* `src/phase_artifacts_v3.py` — `--results_dir`, `--output_dir`, `--models_dir`
  CLI args so V3.1 figures live in `results/v3_1/`.
* `scripts/mega_train_v3_1.sh` — V3.1 one-shot orchestration with
  `--output_suffix _1 --results_dir results/v3_1`, transferring G2→G3→G4→G5
  from `models/v3_1/...`.

### Phase V3.1-3 — Training ✅

~4 h on RTX 5070 Ti. Ammo mode auto-selected = **global** (g1a = g1b ≥-tiebreak).

| Stage | Steps     | Arch            | Transferred     |
|-------|-----------|-----------------|-----------------|
| G2    | 1,000,000 | Attention       | scratch         |
| G3    | 1,500,000 | Attention + GRU | 28 / 32 from G2 |
| G4    | 1,500,000 | Attention + GRU | 32 / 32 from G3 |
| G5    | 2,000,000 | Attention + GRU | 32 / 32 from G4 |

### Phase V3.1-4 — Evaluation ✅ (commit `67a4234`)

10 stochastic + 10 deterministic episodes per learned game; 10 stochastic per
heuristic game; seed `42 + ep_idx`; demos recorded for **all 7 games**
(10 eps × 7 games = 70 MP4s, using `--output_suffix record` so demo re-runs do
not clobber canonical eval JSONs).

| Game | Stochastic      | Deterministic  | Kills A/K  | Failures |
|------|-----------------|----------------|------------|----------|
| G0   | 8.70 ± 9.65     | —              | 7.5 / 4.2  | 3.0      |
| G1a  | 1.70 ± 3.00     | —              | 1.0 / 3.6  | 2.9      |
| G1b  | 1.60 ± 2.76     | —              | 0.9 / 3.6  | 2.9      |
| G2   | 1.80 ± 1.08     | −1.90 ± 1.14   | 4.5 / 0.1  | 2.8      |
| G3   | 4.10 ± 2.17     | 0.60 ± 1.43    | 5.5 / 1.0  | 2.4      |
| G4   | 7.40 ± 3.32     | 0.20 ± 2.71    | 10.0 / 0.1 | 2.7      |
| G5   | **11.20 ± 3.12**| **2.90 ± 2.62**| 12.3 / 0.7 | 1.8      |

**Ship-gates** (all PASS): g1a/g1b failures > 0; ≥1 learned game with failures > 0;
G3≥G2; G4≥G3; G5≥G4.

**Three V3.1 findings:**
1. **Monotonic G2→G5** (1.80 → 4.10 → 7.40 → 11.20). Peak is G5, not G3 as
   in V3, because under actual pressure the pragmatic override pays off.
2. **Failures are now nonzero** (1.8–3.0/ep), so the `-1.0 × failures`
   reward term is part of the learning signal rather than a dead branch.
   G5 minimises failures (1.8) while maximising score — genuine offense /
   defense trade-off.
3. **Knights rejoin the fight.** V3 G4/G5 produced 0.0 knight-kills/ep. V3.1
   with wider lock radius (0.25W) and higher stamina (150) yields 0.1→1.0
   knight-kills/ep across G3–G5.

### Phase V3.1-5 — Documentation ✅ (commit C)

* `README.md` rewrite (V3.1 as canonical, V3 as historical sensitivity baseline).
* `docs/report.tex` appended with §7 V3.1 Addendum (new table, figures pointing
  to `../results/v3_1/`, updated discussion).
* `plan.md` this section.
* V3 artifacts (`models/v3/`, `results/v3/`, `results/tensorboard_v3/`) remain
  intact and are referenced from the addendum as a controlled low-pressure
  point of comparison.

---

## V3 — Structured Teamwork (superseded by V3.1, preserved)

> Branch: `v3-overhaul` · Reference spec: `docs/v3_implementation_plan.md`

**Thesis.** Performance in a cooperative KAZ team comes from *structure*
(roles, target locks, pragmatic override) rather than reward magnitude. We
evaluate four coordination primitives in a 7-game ablation.

### Phase V3-1 — Pre-flight & smoke ✅
* `scripts/_preflight_v3.py` — verified KAZ internals (FPS 15, obs (27,5), action
  mapping: ext 4=attack, ext 5=noop; corrected plan constants).
* `scripts/_smoke_v3.py` — instantiates every `game_level` ∈ `{g0,g1a,g1b,g2,g3,g4,g5}`,
  confirms obs_dim = 140 and that 5 random steps complete without exceptions.

### Phase V3-2 — Implementation ✅
Commit 1: `feat(v3): wrapper, heuristic policies, model extras, training & eval scripts` (`c75467f`).
Commit 2: `fix(v3): pragmatic-override counter inflation + direction-aware patrol hard mask` (`b2e2369`).

* `src/wrappers/kaz_wrapper_v3.py` — 7-game config table, ammo & stamina pools,
  archer-immobile mode, role assignment, target-lock acquisition, G5 pragmatic
  override, end-zone crossing detection, 5-dim observation extras.
* `src/policies/heuristic.py` — scripted aim-and-fire / walk-and-attack for G0/G1a/G1b.
* `src/models/mappo_net.py` — added `extra_dim` parameter; attention/attention+GRU
  branches project extras via `nn.Linear(extra_dim, hidden_dim)` and add to pooled
  entity vector.
* `src/train_v3.py` — PPO loop with new TensorBoard scalars (`metric/score`,
  `metric/failures`, `metric/ammo_pct_remaining`, `metric/stamina_pct_remaining`,
  `metric/archer_kills`, `metric/knight_kills`, etc.), transfer-learning loader.
* `src/evaluate_v3.py` — all 7 games, per-episode seed = `args.seed + ep_idx` so
  zombie patterns match across games; stochastic + `--deterministic` modes.
* `src/phase_artifacts_v3.py` — score evolution, metric breakdown, attention
  saliency heatmaps (G3 vs G5), side-by-side demo stitcher, ablation table.
* `scripts/mega_train_v3.sh` — orchestration script (heuristic evals → ammo-mode
  selection → train G2 → transfer G3 → transfer G4 → transfer G5).

### Phase V3-3 — Training ✅

~3.5 h on RTX 5070 Ti (~480 SPS). Ammo-mode auto-selected = **global** (G1a ≥ G1b tied;
deterministic tiebreak).

| Stage | Steps     | Params  | Arch            | Transferred    |
|-------|-----------|---------|-----------------|----------------|
| G2    | 1,000,000 | 432,265 | Attention       | scratch        |
| G3    | 1,500,000 | 827,017 | Attention + GRU | 28 / 32 from G2 (4 GRU tensors new) |
| G4    | 1,500,000 | 827,017 | Attention + GRU | 32 / 32 from G3 |
| G5    | 2,000,000 | 827,017 | Attention + GRU | 32 / 32 from G4 |

### Phase V3-4 — Evaluation ✅

10 episodes per game (+ 10 deterministic for G2–G5) + 3-episode video recordings for
G0 and G5. Seed convention: `42 + ep_idx` so every game's episode-n uses the same
zombie spawn pattern.

| Game | Stochastic      | Deterministic | Kills A/K | Failures |
|------|-----------------|---------------|-----------|----------|
| G0   | **10.10 ± 4.11**| —             | 4.5 / 5.6 | 0.0      |
| G1a  | 3.70 ± 1.19     | —             | 0.6 / 3.1 | 0.0      |
| G1b  | 3.70 ± 1.19     | —             | 0.6 / 3.1 | 0.0      |
| G2   | 2.40 ± 0.66     | 0.80 ± 0.75   | 2.3 / 0.1 | 0.0      |
| G3   | **4.70 ± 1.49** | 2.40 ± 1.28   | 4.4 / 0.3 | 0.0      |
| G4   | 3.90 ± 0.70     | **2.70 ± 1.27** | 3.9 / 0.0 | 0.0    |
| G5   | 3.50 ± 1.28     | 2.60 ± 0.92   | 3.5 / 0.0 | 0.0      |

**Three findings:**
1. Roles (G3) are the single biggest structural win: **+96 %** over G2 with no
   extra reward engineering.
2. Deterministic scores increase G2→G4 (0.80 → 2.70), reversing the V1/V2
   deterministic-passivity pattern — structure hardens the argmax mode.
3. Tight knight locks (G4/G5) produce **knight passivity**: archers do all the
   kills while knights attack ~0 times/ep. Section 5.3 of the report discusses
   remediation (wider lock radius / lower knight stamina cost / learned role bonus).

### Phase V3-5 — Documentation ✅
* `README.md` rewrite (V3 narrative).
* `docs/report.tex` + `docs/report.pdf` — full academic write-up of V3.
* `docs/presentation_guide.md` — 15-min slide-by-slide deck.
* `results/v3/ablation_table.md` — auto-generated markdown table.
* V1/V2 plan sections below preserved unchanged.

---

## V2 — Death Penalty Iteration (superseded by V3)

> Branch: `v2-death-penalty` · Historical; see `docs/v2_death_penalty_plan.md`.

* `KAZWrapper` gained a `death_penalty` parameter (default 2.0) applied on agent
  termination, threaded through training, evaluation, and `phase5.py`.
* All 5 games retrained with `--death_penalty 2.0`.
* Key finding: death penalty alone does **not** fix the MLP passivity collapse
  (G2/G3 deterministic kills remain 0/ep). Only G5 (attention + GRU) converts the
  survival signal into active behaviour (raw kill density 0.00373, +25 % vs V1).

| Game | V1 Raw Kill Density | V2 Raw Kill Density | Δ       |
|------|---------------------|---------------------|---------|
| G1   | 0.01089             | 0.01051             | −3.5 %  |
| G2   | 0.00431             | 0.00308             | −29 %   |
| G3   | 0.00279             | 0.00297             | +6.5 %  |
| G4   | 0.00361             | 0.00298             | −17 %   |
| G5   | 0.00299             | **0.00373**         | **+25 %** |

Artifacts preserved in `models/game{1..5}/`, `results/game*_eval_results*.json`,
`results/final_ablation_table.md`, `results/phase4_observations.md`.

---

## V1 — 5-Game Ablation "Evolution of Fire Discipline" (superseded by V3)

> Branch: `main` · Historical.

### Phase 1 — G1 Villain Baseline ✅
Default KAZ with kill reward. "Greedy Soldier": charging zombies blindly,
overlapping paths, zero coordination. 502 k steps / 409 episodes. Mean return
5.80 ± 2.94; raw kill density 0.01089.

### Phase 2 — G2/G3 Resource Scarcity ✅
* G2 (+15 arrow/ep ammo, −0.5 dry-fire penalty): 504 k steps / 676 ep. The "Risk
  Avoider." Archer drops to 0.15 % shoot rate under deterministic eval.
* G3 (+0.01 move / +0.05 attack stamina): 500 k steps / 697 ep. "Fully Passive."
  Knight drops attack rate to 0 %.
Key finding: penalties too strong relative to kill reward → agents learn avoidance
rather than efficiency.

### Phase 3 — G4 Altruistic Hero ✅
* 60/40 team reward + entity self-attention encoder. 508 k steps / 706 ep.
  430 k params. "Recovering Cooperator": kill density reverses downward trend
  (+29 % vs G3); team reward flows (preservation +0.68). Deterministic still
  passive (0 kills/ep).

### Phase 4 — G5 Fire Discipline ✅
* Gaussian fog (σ=0.3) + Attention + GRU (825 481 params). 1.5 M steps, transfer-
  loaded 26 / 26 tensors from G4. Deterministic attack rate: **69.9 %** (vs 0 % in
  G4). Both aggression and preservation positive simultaneously.
* Diagnosed *deterministic passivity* as a previously under-described MARL failure
  mode; GRU temporal memory resolves it.

### Phase 5 — Evaluation & Explainability ✅
Ablation table, collapse graph (G1 vs G5 at 0.7×/1×/1.4×/2× spawn pressure),
saliency comparison (MLP gradient vs attention heatmap), 30 s side-by-side demo
video. Script: `src/phase5.py`.

---

## Artifacts preserved as reference

`phase{1..4}_plan.md`, `v2_death_penalty_plan.md`, `models/game{1..5}/`,
`results/game*_eval_results*.json`, `results/final_ablation_table.md`,
`results/kill_density_evolution.png`, `results/collapse_graph.png`,
`results/saliency_comparison.png`, `results/demo_sidebyside.mp4`.
