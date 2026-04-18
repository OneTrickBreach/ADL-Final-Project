## Structured Teamwork in Multi-Agent PPO
### ADL CS7180 Final Project — V3 Overhaul: A 7-Game Study of Coordination Primitives

**Team:** Ishan Biswas, Elizabeth Coquillette, Nishant Suresh
**Branch of record:** `v3-overhaul`

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

> **Relationship to V1/V2.** Earlier iterations on `main` explored the
> "deterministic-passivity" failure mode from stacking penalty constraints. V3
> replaces that narrative with a cleaner positive-reinforcement story: observe
> how much performance comes from *structure* once constraints are fixed and
> shared across all games.

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

## 3. Headline results

*10 evaluation episodes per game, seed 42+ep_idx, zombies identical across games.*

| Game | Stochastic score | Deterministic score | %Ammo | %Stamina | Kills (A / K) | Failures |
|------|------------------|---------------------|-------|----------|---------------|----------|
| G0   | **10.10 ± 4.11** | —                   | 0.00  | 0.00     | 4.5 / 5.6     | 0.0      |
| G1a  | 3.70 ± 1.19      | —                   | 0.66  | 0.99     | 0.6 / 3.1     | 0.0      |
| G1b  | 3.70 ± 1.19      | —                   | 0.64  | 0.99     | 0.6 / 3.1     | 0.0      |
| G2   | 2.40 ± 0.66      | 0.80 ± 0.75         | 1.00  | 0.88     | 2.3 / 0.1     | 0.0      |
| G3   | **4.70 ± 1.49**  | 2.40 ± 1.28         | 1.00  | 0.98     | 4.4 / 0.3     | 0.0      |
| G4   | 3.90 ± 0.70      | **2.70 ± 1.27**     | 1.00  | 0.92     | 3.9 / 0.0     | 0.0      |
| G5   | 3.50 ± 1.28      | 2.60 ± 0.92         | 0.93  | 0.96     | 3.5 / 0.0     | 0.0      |

**Three take-aways:**

- **Peak learned performance is G3 (roles + GRU)**, not G5. Adding a role structure
  lifts the score by **+96 %** over G2 with no extra reward engineering.
- **Deterministic policies improve across the progression** (G2→G4: **0.80 → 2.70**),
  i.e. constraints force the policy's argmax mode to commit to attacks rather than
  sample them stochastically. This reverses the V1/V2 "deterministic passivity" pattern.
- **Knight passivity emerges under tight locks.** In G4/G5 the knight lock
  radius + forward-only role turns the knight into a near-silent partner (≈0 attacks)
  while the archer does all of the work. Section 5.3 of the report discusses this as
  a limitation and suggests remediations for future work.

---

## 4. Quick start

```bash
# 1. Setup (Python 3.10–3.12)
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or cpu/mps
pip install "pettingzoo[butterfly]" gymnasium supersuit tensorboard opencv-python matplotlib numpy

# 2. Smoke-test the 7 games (<10 s)
./.venv/bin/python scripts/_smoke_v3.py

# 3. End-to-end training (~3–4 h on RTX 5070 Ti)
bash scripts/mega_train_v3.sh

# 4. Evaluation pipeline (Cascade documented in docs/v3_implementation_plan.md §6)
for g in g0 g1a g1b; do ./.venv/bin/python src/evaluate_v3.py --game $g --episodes 10 --seed 42; done
for g in g2 g3 g4 g5; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3/$g/final.pt --episodes 10 --seed 42
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3/$g/final.pt --episodes 10 --seed 42 --deterministic
done

# 5. Demos and figures
./.venv/bin/python src/evaluate_v3.py --game g0 --episodes 3 --seed 42 --record
./.venv/bin/python src/evaluate_v3.py --game g5 --checkpoint models/v3/g5/final.pt --episodes 3 --seed 42 --record
./.venv/bin/python src/phase_artifacts_v3.py
```

TensorBoard: `tensorboard --logdir results/tensorboard_v3`.

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
│   ├── mega_train_v3.sh              # Full training orchestration
│   ├── _preflight_v3.py              # Sanity-check KAZ internals
│   └── _smoke_v3.py                  # 7-game wrapper smoke test
├── models/
│   ├── game{1..5}/                   # V1/V2 checkpoints (historical)
│   └── v3/{g2,g3,g4,g5}/final.pt     # V3 checkpoints
├── results/
│   ├── tensorboard/                  # V1/V2 training curves
│   ├── tensorboard_v3/               # V3 training curves
│   ├── v3/
│   │   ├── ablation_table.md         # Generated markdown ablation table
│   │   ├── score_evolution.png       # Bar chart: G0 → G5 mean score
│   │   ├── metric_breakdown.png      # 2×2 grid: ammo/stamina/kills/attacks
│   │   ├── saliency_v3.png           # Attention heatmaps G3 vs G5
│   │   ├── demo_sidebyside_v3.mp4    # G0 heuristic vs G5 learned, same seed
│   │   ├── g*_eval_results*.json     # Per-game eval JSONs (stoch + det)
│   │   ├── g*_demo/                  # Recorded gameplay MP4s
│   │   └── ammo_mode.txt             # Auto-selected ammo pool mode
├── docs/
│   ├── report.pdf / report.tex       # V3 report (LaTeX)
│   ├── references.bib
│   ├── v3_implementation_plan.md     # Authoritative implementation plan for V3
│   ├── v3_psudoplan.txt              # Group-approved design sketch (source of truth)
│   ├── presentation_guide.md         # V3 slide-by-slide talk guide
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

| Milestone                                      | Status |
|------------------------------------------------|--------|
| V1 — 5-game ablation (G1–G5)                   | ✅ preserved on `main` for reference |
| V2 — death-penalty iteration                   | ✅ preserved on `v2-death-penalty`   |
| **V3 — structured teamwork (G0–G5)**           | ✅ **current `v3-overhaul` branch**  |
| Training pipeline (6M steps total)             | ✅ complete                           |
| Evaluation pipeline (70 episodes, stoch + det) | ✅ complete                           |
| Phase artifacts (4 figures + demo mp4)         | ✅ complete                           |
| Documentation (report, README, plan, slides)   | ✅ complete                           |
