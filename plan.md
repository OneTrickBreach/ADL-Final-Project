## ADL Final Project — Roadmap

Live ledger for all three iterations of this study. Most recent work is V3
("Structured Teamwork"); V1 and V2 are preserved unchanged for reference.

---

## V3 — Structured Teamwork (current)

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
