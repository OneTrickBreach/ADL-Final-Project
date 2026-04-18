# Structured Teamwork in Multi-Agent PPO
### A Seven-Game Study of Coordination Primitives on Knights-Archers-Zombies

**Team:** Ishan Biswas, Elizabeth Coquillette, Nishant Suresh
**Course:** CS7180 Advanced Deep Learning

---

## 1. What this project does

A controlled seven-game ablation on PettingZoo's **Knights-Archers-Zombies (KAZ)** that
isolates the contribution of four coordination primitives to cooperative MARL
performance:

1. **Ammo / stamina discipline** — forces agents to value each action.
2. **Role assignment** (forward knight vs. end-zone patrol knight) — divides labour spatially.
3. **Target locks** — prevents duplicate targeting between classes.
4. **Pragmatic override** — reintroduces flexibility when strict rules would cost the team.

All seven games share identical episode length, spawn rate, and per-episode random
seeds so that score differences isolate the effect of each primitive. Games G0–G1b
are **scripted heuristic baselines**; G2–G5 are trained with shared-parameter
**MAPPO** (PPO + GAE) using an entity self-attention encoder (G2) or attention +
GRU (G3–G5) with transfer-learning between adjacent games.

---

## 2. The 7 games

| Game | Name                  | Policy       | Architecture    | New primitive added                                                  |
|------|-----------------------|--------------|-----------------|----------------------------------------------------------------------|
| G0   | Unrestricted          | heuristic    | —               | baseline; no ammo / stamina caps                                     |
| G1a  | Global-pool ammo      | heuristic    | —               | shared ammo + knight stamina; archers immobile                       |
| G1b  | Individual-pool ammo  | heuristic    | —               | per-archer ammo + knight stamina; archers immobile                   |
| G2   | Basic teamwork        | MAPPO        | Attention       | ammo discipline learned (auto-select best ammo mode from G1a vs G1b) |
| G3   | Role assignment       | MAPPO + TL   | Attention + GRU | 1 forward knight + 1 end-zone patrol knight                          |
| G4   | Target locks          | MAPPO + TL   | Attention + GRU | knight lock radius; archer skips knight-locked zombies               |
| G5   | Pragmatic override    | MAPPO + TL   | Attention + GRU | archer overrides knight lock if zombie will cross first              |

*TL = transfer-learning from previous game's final checkpoint.*

Every episode lasts **30 seconds (450 steps @ 15 FPS)**. **Score = kills − failures**,
where a *failure* is a zombie that crosses the bottom of the screen *or* an agent
killed by zombie contact (both indicate the defence line was breached).

---

## 3. Headline results

*10 evaluation episodes per game (G2–G5 additionally with 10 deterministic/argmax
episodes). Seed convention: episode `i` uses seed `42 + i` identically across all
games, so zombie spawn patterns are directly comparable.*

| Game | Stochastic score | Deterministic score | %Ammo | %Stamina | Kills (A / K) | Failures |
|------|------------------|---------------------|-------|----------|---------------|----------|
| G0   | 8.70 ± 9.65      | —                   | 0.00  | 0.00     | 7.5 / 4.2     | 3.0      |
| G1a  | 1.70 ± 3.00      | —                   | 0.60  | 0.55     | 1.0 / 3.6     | 2.9      |
| G1b  | 1.60 ± 2.76      | —                   | 0.60  | 0.55     | 0.9 / 3.6     | 2.9      |
| G2   | 1.80 ± 1.08      | −1.90 ± 1.14        | 0.97  | 0.86     | 4.5 / 0.1     | 2.8      |
| G3   | 4.10 ± 2.17      | 0.60 ± 1.43         | 0.96  | 0.85     | 5.5 / 1.0     | 2.4      |
| G4   | 7.40 ± 3.32      | 0.20 ± 2.71         | 0.90  | 0.93     | 10.0 / 0.1    | 2.7      |
| **G5** | **11.20 ± 3.12** | **2.90 ± 2.62**   | 0.88  | 0.81     | 12.3 / 0.7    | 1.8      |

**Three take-aways:**

- **Strictly monotonic stochastic progression G2 → G5** (1.80 → 4.10 → 7.40 → 11.20;
  each primitive is additive). The deterministic progression climbs from strongly
  negative to strongly positive along the same sequence
  (−1.90 → 0.60 → 0.20 → 2.90); G3 and G4 are tied within noise and **G5 is the
  peak on both metrics**, outperforming the unrestricted heuristic ceiling G0
  (8.70) by **+29%** on stochastic score.
- **The failure signal is active.** Every game reports 1.8–3.0 failures/episode, so
  the failure penalty `−1.0 × failures_this_step` is a live training signal (not a
  vestigial term). G5 *minimises* failures (1.8) while *maximising* score — the
  pragmatic override is performing a genuine offence/defence trade-off.
- **Knights stay active through G5.** With a lock radius of 0.25 W (320 px) and a
  stamina pool of 150 actions, knights continue to attack and kill zombies even
  under the tight G4 / G5 coordination rules (0.1–1.0 kills/ep).

All 70 stochastic + 40 deterministic evaluation episodes, plus 70 recorded demo
MP4s (10 per game) and 4 figures, live in `results/v3_1/`.

---

## 4. Quick start

```bash
# 1. Setup (Python 3.10–3.12)
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or cpu/mps
pip install "pettingzoo[butterfly]" gymnasium supersuit tensorboard opencv-python matplotlib numpy

# 2. End-to-end training (~4 h on RTX 5070 Ti)
bash scripts/mega_train_v3_1.sh

# 3. Evaluation pipeline
RESULTS=results/v3_1
for g in g0 g1a g1b; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --episodes 10 --seed 42 --results_dir $RESULTS
done
for g in g2 g3 g4 g5; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3_1/$g/final.pt \
      --episodes 10 --seed 42 --results_dir $RESULTS
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3_1/$g/final.pt \
      --episodes 10 --seed 42 --deterministic --results_dir $RESULTS
done

# 4. Demo videos (10 eps/game, suffix avoids clobbering canonical eval JSONs)
for g in g0 g1a g1b; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --episodes 10 --seed 42 \
      --record --results_dir $RESULTS --output_suffix record
done
for g in g2 g3 g4 g5; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3_1/$g/final.pt \
      --episodes 10 --seed 42 --record --results_dir $RESULTS --output_suffix record
done

# 5. Figures, ablation table, side-by-side demo
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
│   ├── models/mappo_net.py           # MAPPO policy/value net (Attn / Attn+GRU with extras branch)
│   ├── wrappers/kaz_wrapper_v3.py    # 7-game config, locks, pragmatic override, failure detector
│   ├── policies/heuristic.py         # Scripted policy for G0 / G1a / G1b
│   ├── train_v3.py                   # MAPPO training (Attention / Attention+GRU)
│   ├── evaluate_v3.py                # Evaluator (all 7 games, fixed-seed-per-episode)
│   └── phase_artifacts_v3.py         # Score evolution, metric breakdown, saliency, side-by-side demo
├── scripts/
│   └── mega_train_v3_1.sh            # One-shot training orchestration
├── models/v3_1/{g2,g3,g4,g5}/final.pt
├── results/
│   ├── tensorboard_v3_1/             # Training curves
│   └── v3_1/
│       ├── ablation_table.md
│       ├── score_evolution.png, metric_breakdown.png, saliency_v3.png
│       ├── demo_sidebyside_v3.mp4    # G0 heuristic vs G5 learned, same seed
│       ├── g*_eval_results.json      # Stochastic eval (10 eps)
│       ├── g*_eval_results_det.json  # Deterministic eval (G2–G5)
│       ├── g*_eval_results_record.json
│       ├── g*_demo/episode_{1..10}_seed{42..51}.mp4
│       └── ammo_mode.txt             # Auto-selected ammo pool mode
├── docs/
│   ├── report.pdf / report.tex       # Academic paper
│   ├── references.bib
│   ├── design.md                     # Full design + reproducibility spec
│   └── presentation_guide.md         # 15-minute slide-by-slide talk guide
├── plan.md                           # Milestone ledger
├── rules.md                          # Execution rules
└── requirements.txt
```

---

## 7. Documentation

| Document                               | Purpose                                                  |
|----------------------------------------|----------------------------------------------------------|
| `docs/report.pdf` / `docs/report.tex`  | Academic paper (methods, results, discussion)            |
| `docs/design.md`                       | Reproducible implementation + evaluation spec            |
| `docs/presentation_guide.md`           | 15-min slide-by-slide talk                               |
| `results/v3_1/ablation_table.md`       | Auto-generated markdown ablation table (7 games)         |
| `plan.md`                              | Milestone ledger                                         |
| `rules.md`                             | Execution rules (always use `./.venv/bin/python`)        |

---

## 8. Prior exploration (for context)

An earlier iteration of this project (preserved on the `legacy` branch) ran a
5-game ablation on the same KAZ environment but shaped cooperation through
stacked reward penalties (ammo cost, stamina cost, fog-of-war, 60 / 40 team
blend) rather than through structural primitives. Three shortcomings in that
iteration motivated the redesign captured in this branch:

- **No heuristic baselines.** Without G0 / G1a / G1b, there was no way to tell
  whether a measured kill density was good or bad relative to a trivial
  scripted policy.
- **No live failure signal.** The environment never produced a zombie crossing
  during evaluation, so the defence-penalty term in the reward was inert at
  training time — a bug we later traced to the failure detector.
- **Deterministic passivity.** Under penalty stacking the argmax policy
  collapsed to near-zero activity even when the stochastic policy still fired,
  which made the headline results hard to defend under strict evaluation.

Every design choice in the current project (heuristic baselines, position-plus-breach
failure detector, structural primitives with minimal shaping, identical
per-episode seeds) is a direct response to one of those shortcomings.

---

## 9. Status

| Milestone                                      | Status |
|------------------------------------------------|--------|
| Wrapper, heuristic policies, model with extras | ✅     |
| Training pipeline (6M steps total)             | ✅     |
| Evaluation pipeline (70 stoch + 40 det eps)    | ✅     |
| Demo recordings (10 eps × 7 games)             | ✅     |
| Phase artifacts (4 figures + side-by-side mp4) | ✅     |
| Documentation (report, design, plan, slides)   | ✅     |
