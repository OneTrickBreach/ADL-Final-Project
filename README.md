## Structured Teamwork in Multi-Agent PPO
### ADL CS7180 Final Project — V3.1: A 7-Game Study of Coordination Primitives Under Pressure

**Team:** Ishan Biswas, Elizabeth Coquillette, Nishant Suresh
**Branch of record:** `v3-overhaul`

> **V3.1 supersedes V3.** The original V3 run (`models/v3/`, `results/v3/`)
> used a buggy failure detector that undercounted line crosses and parameter
> defaults that made the environment too easy for the failure penalty to fire
> (see `docs/v3_1_preflight_findings.md`, `docs/v3_1_retrain_plan_fixed.md`).
> V3.1 rewrites the detector, retunes six pressure parameters, and retrains
> G2–G5. All V3 artifacts are preserved for the sensitivity appendix. Unless
> stated otherwise, **numbers and figures below are V3.1** (`models/v3_1/`,
> `results/v3_1/`).

---

## 1. What this project does

A controlled 7-game ablation on PettingZoo's **Knights-Archers-Zombies (KAZ)** that
isolates the contribution of four coordination primitives to cooperative MARL
performance:

1. **Ammo / stamina discipline** — forces agents to value each action.
2. **Role assignment** (forward knight vs. patrol knight) — divides labour spatially.
3. **Target locks** — prevents duplicate targeting between classes.
4. **Pragmatic override** — reintroduces flexibility when rules would cost the team.

All seven games are evaluated on **identical zombie patterns** (same seed per episode)
so that score differences isolate the effect of each primitive. Games G0–G1b are
**scripted heuristic baselines**; G2–G5 are trained with shared-parameter **MAPPO**
(PPO + GAE) using an entity self-attention encoder (G2) or attention + GRU (G3–G5).

> **Relationship to V1/V2/V3.** Earlier iterations on `main` explored the
> "deterministic-passivity" failure mode from stacking penalty constraints.
> V3 on `v3-overhaul` introduced the 7-game structure but under mild pressure
> (spawn_rate 20, max_zombies 10) where failures never fired. V3.1 keeps the
> same structure + curriculum but fixes the detector (line-cross on
> `centery + agent-breach via sprite-list delta) and increases pressure
> (spawn_rate 8, max_zombies 20, lock radius 0.25W, ammo 60/30, stamina 150)
> so the failure penalty is an active training signal.

---

## 2. The 7 games

| Game | Name                  | Learner?     | Arch            | New primitive added                                                  |
|------|-----------------------|--------------|-----------------|----------------------------------------------------------------------|
| G0   | Unrestricted          | heuristic    | —               | baseline; no ammo/stamina                                            |
| G1a  | Global-pool ammo      | heuristic    | —               | shared ammo, knight stamina, archers immobile                        |
| G1b  | Individual-pool ammo  | heuristic    | —               | per-archer ammo, stamina, archers immobile                           |
| G2   | Basic teamwork        | MAPPO        | Attention       | ammo-discipline learned (best of G1a/G1b mode, selected automatically) |
| G3   | Role assignment       | MAPPO + TL   | Attention + GRU | 1 forward knight, 1 end-zone patrol knight                           |
| G4   | Target locks          | MAPPO + TL   | Attention + GRU | knight lock radius; archer skips knight-locked zombies               |
| G5   | Pragmatic override    | MAPPO + TL   | Attention + GRU | archer overrides knight lock if zombie will cross first              |

*TL = transfer-learning from previous game's final checkpoint.*

Every episode lasts **30 seconds (450 steps @ 15 FPS)**; we report **score = kills − failures**,
where a *failure* is a zombie that reaches the bottom of the screen.

---

## 3. Headline results (V3.1)

*10 evaluation episodes per game, seed 42+ep_idx, zombies identical across games.*

| Game | Stochastic score | Deterministic score | %Ammo | %Stamina | Kills (A / K) | Failures |
|------|------------------|---------------------|-------|----------|---------------|----------|
| G0   | 8.70 ± 9.65      | —                   | 0.00  | 0.00     | 7.5 / 4.2     | 3.0      |
| G1a  | 1.70 ± 3.00      | —                   | 0.60  | 0.55     | 1.0 / 3.6     | 2.9      |
| G1b  | 1.60 ± 2.76      | —                   | 0.60  | 0.55     | 0.9 / 3.6     | 2.9      |
| G2   | 1.80 ± 1.08      | −1.90 ± 1.14        | 0.97  | 0.86     | 4.5 / 0.1     | 2.8      |
| G3   | 4.10 ± 2.17      | 0.60 ± 1.43         | 0.96  | 0.85     | 5.5 / 1.0     | 2.4      |
| G4   | 7.40 ± 3.32      | 0.20 ± 2.71         | 0.90  | 0.93     | 10.0 / 0.1    | 2.7      |
| G5   | **11.20 ± 3.12** | **2.90 ± 2.62**     | 0.88  | 0.81     | 12.3 / 0.7    | 1.8      |

**Three take-aways (V3.1):**

- **Monotonic progression G2 → G5.** Stochastic score climbs
  1.80 → 4.10 → 7.40 → 11.20 (each primitive is additive). Deterministic
  tracks the same shape (−1.90 → 0.60 → 0.20 → 2.90). Peak is G5, not G3
  as in V3.
- **Failures are actually used.** Unlike V3 where all seven games reported 0
  failures, V3.1 has 1.8–3.0 failures/episode across the progression, so the
  failure penalty is now part of the training signal. G5 achieves the lowest
  failure count (1.8) while also scoring the highest, i.e. the pragmatic
  override is genuinely trading off offense and defense.
- **Knights participate at G5.** V3 produced near-silent knights at G4/G5
  (0.0 kills/ep). V3.1 with a wider lock radius (0.25W vs 0.15W) and higher
  stamina (150 vs 100) keeps knights active through G5 (0.7 kills/ep at G5).

**Ship-gates** (see `docs/v3_1_retrain_plan_fixed.md` §7) all PASS:
g1a/g1b failures > 0 ✓, ≥1 learned game with failures > 0 ✓, G3≥G2 ✓,
G4≥G3 ✓, G5≥G4 ✓.

All 70 stochastic + 40 deterministic evaluation episodes plus 70 recorded
demo MP4s (10 per game) are in `results/v3_1/`.

---

## 4. Quick start

```bash
# 1. Setup (Python 3.10–3.12)
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or cpu/mps
pip install "pettingzoo[butterfly]" gymnasium supersuit tensorboard opencv-python matplotlib numpy

# 2. Smoke-test the 7 games (<10 s)
./.venv/bin/python scripts/_smoke_v3.py

# 3. End-to-end training (~4 h on RTX 5070 Ti)
bash scripts/mega_train_v3_1.sh      # V3.1 (current canonical run)
# bash scripts/mega_train_v3.sh      # V3 (historical, lower-pressure defaults)

# 4. Evaluation pipeline
RESULTS=results/v3_1
for g in g0 g1a g1b; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --episodes 10 --seed 42 --results_dir $RESULTS
done
for g in g2 g3 g4 g5; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3_1/$g/final.pt --episodes 10 --seed 42 --results_dir $RESULTS
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3_1/$g/final.pt --episodes 10 --seed 42 --deterministic --results_dir $RESULTS
done

# 5. Demo videos (10 eps/game across all 7 games; use --output_suffix record to
#    avoid clobbering the canonical JSONs above)
for g in g0 g1a g1b; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --episodes 10 --seed 42 \
      --record --results_dir $RESULTS --output_suffix record
done
for g in g2 g3 g4 g5; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3_1/$g/final.pt \
      --episodes 10 --seed 42 --record --results_dir $RESULTS --output_suffix record
done

# 6. Figures, ablation table, side-by-side demo
./.venv/bin/python src/phase_artifacts_v3.py \
    --results_dir results/v3_1 --output_dir results/v3_1 --models_dir models/v3_1
```

TensorBoard: `tensorboard --logdir results/tensorboard_v3_1`.

---

## 5. Platform compatibility

The codebase auto-detects CUDA → MPS → CPU at runtime. Tested on:

| Platform              | Device | Notes                                   |
|-----------------------|--------|-----------------------------------------|
| Linux / NVIDIA GPU    | CUDA   | RTX 5070 Ti, ~480 SPS, full run ~3.5 h  |
| macOS (Apple Silicon) | MPS    | works out of the box                    |
| CPU-only              | CPU    | ~10–20× slower                          |

---

## 6. Repository layout

```
ADLProject2/
├── src/
│   ├── models/mappo_net.py           # MAPPO policy/value net (MLP / Attn / Attn+GRU, extra_dim)
│   ├── wrappers/
│   │   ├── kaz_wrapper.py            # V1/V2 wrapper (kept for reference)
│   │   └── kaz_wrapper_v3.py         # V3 wrapper: 7-game config, locks, pragmatic override
│   ├── policies/heuristic.py         # Scripted policy for G0 / G1a / G1b
│   ├── train.py, evaluate.py, phase5.py  # V1/V2 scripts (preserved)
│   ├── train_v3.py                   # V3 MAPPO training (attention / attention+GRU)
│   ├── evaluate_v3.py                # V3 evaluator (all 7 games, fixed-seed-per-episode)
│   └── phase_artifacts_v3.py         # Score evolution, metric breakdown, saliency, side-by-side demo
├── scripts/
│   ├── mega_train_v3.sh              # V3 orchestration (historical)
│   ├── mega_train_v3_1.sh            # V3.1 orchestration (current)
│   ├── _preflight_v3.py              # Sanity-check KAZ internals
│   └── _smoke_v3.py                  # 7-game wrapper smoke test
├── models/
│   ├── game{1..5}/                   # V1/V2 checkpoints (historical)
│   ├── v3/{g2,g3,g4,g5}/final.pt     # V3 checkpoints (historical, lower pressure)
│   └── v3_1/{g2,g3,g4,g5}/final.pt   # V3.1 checkpoints (current canonical)
├── results/
│   ├── tensorboard/                  # V1/V2 training curves
│   ├── tensorboard_v3/               # V3 training curves (historical)
│   ├── tensorboard_v3_1/             # V3.1 training curves (current)
│   ├── v3/                           # V3 artifacts (preserved for sensitivity)
│   └── v3_1/                         # V3.1 artifacts (current canonical)
│       ├── ablation_table.md
│       ├── score_evolution.png, metric_breakdown.png, saliency_v3.png
│       ├── demo_sidebyside_v3.mp4    # G0 heuristic vs G5 learned, same seed
│       ├── g*_eval_results.json      # Canonical stochastic eval (10 eps)
│       ├── g*_eval_results_det.json  # Canonical deterministic eval (G2–G5)
│       ├── g*_eval_results_record.json  # Re-run alongside video recording
│       ├── g*_demo/episode_{1..10}_seed{42..51}.mp4
│       └── ammo_mode.txt             # Auto-selected ammo pool mode
├── docs/
│   ├── report.pdf / report.tex       # V3 report + V3.1 addendum
│   ├── references.bib
│   ├── v3_implementation_plan.md     # Authoritative implementation plan for V3
│   ├── v3_1_preflight_findings.md    # Why V3's detector was wrong
│   ├── v3_1_retrain_plan.md          # Initial V3.1 plan (parameter-only)
│   ├── v3_1_retrain_plan_fixed.md    # Authoritative V3.1 plan (detector rewrite)
│   ├── v3_psudoplan.txt              # Group-approved design sketch
│   ├── presentation_guide.md         # Slide-by-slide talk guide
│   └── phase{1..4}_plan.md, v2_death_penalty_plan.md  # V1/V2 plans (historical)
├── plan.md                           # Roadmap: V1, V2, V3 milestones
├── rules.md                          # Execution rules
└── requirements.txt
```

---

## 7. Documentation

| Document                                      | Purpose                                         |
|-----------------------------------------------|-------------------------------------------------|
| `docs/report.pdf` / `docs/report.tex`         | Academic paper covering V3 (methods, results)   |
| `docs/v3_implementation_plan.md`              | Reproducible spec (§4 code, §6 eval, §7 docs)   |
| `docs/presentation_guide.md`                  | 15-min talk with slide-by-slide notes           |
| `results/v3/ablation_table.md`                | Auto-generated markdown table (all 7 games)     |
| `plan.md`                                     | V1 / V2 / V3 milestone ledger                   |
| `rules.md`                                    | Execution rules (always use `./.venv/bin/python`)|

---

## 8. Status

| Milestone                                         | Status |
|---------------------------------------------------|--------|
| V1 — 5-game ablation (G1–G5)                      | ✅ preserved on `main` for reference |
| V2 — death-penalty iteration                      | ✅ preserved on `v2-death-penalty`   |
| V3 — structured teamwork (G0–G5, low pressure)    | ✅ preserved under `v3/` and `models/v3/` |
| **V3.1 — detector rewrite + retune (current)**    | ✅ **current `v3-overhaul` branch**  |
| V3.1 training pipeline (6M steps total)           | ✅ complete                           |
| V3.1 evaluation pipeline (70 stoch + 40 det eps)  | ✅ complete                           |
| V3.1 demo recordings (10 eps × 7 games)           | ✅ complete                           |
| V3.1 phase artifacts (4 figures + demo mp4)       | ✅ complete                           |
| Documentation (report + addendum, README, plan)   | ✅ complete                           |
