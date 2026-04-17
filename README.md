# The Evolution of Fire Discipline
### ADL CS7180 Final Project — 5-Game MARL Ablation Study

**Team:** Ishan Biswas, Elizabeth Coquillette, Nishant Suresh

---

## What This Project Does

A complete multi-agent reinforcement learning study using **MAPPO** on the PettingZoo **Knights, Archers & Zombies (KAZ)** environment. Five games are trained in sequence, each adding exactly one new constraint, to trace how reward shaping and architecture drive the emergence of coordinated behavior — from a mindlessly aggressive baseline to a disciplined tactical unit that operates under resource limits, teammate awareness, and perceptual fog.

**V2 design iteration:** Added a configurable **death penalty** (−2.0 per agent death) to test whether a survival incentive alone could fix the passivity trap. **It cannot** — MLP agents (G2–G3) remain fully passive even with death penalty. Only GRU temporal memory (G5) converts the survival signal into active behavior, confirming that architectural capacity is the bottleneck, not reward magnitude.

**Central finding:** *Deterministic passivity* — a previously under-described MARL failure mode where the stochastic policy attacks but the argmax (deployment) policy is completely passive. Death penalty alone is insufficient; Game 5's GRU temporal memory resolves it, achieving non-zero deterministic kills vs 0 kills in Games 2–4.

---

## Results at a Glance

| Game | Label | Mean Return | Raw Kill Density | Det. Raw Kills | Architecture |
|------|-------|-------------|-----------------|---------------|-------------|
| G1 | Greedy Soldier | 3.98 ± 3.80 | **0.01051** | 0.70/ep | MLP |
| G2 | Risk Avoider | −1.75 ± 0.52 | 0.00308 | **0.05/ep** ← passive | MLP |
| G3 | Fully Passive | −1.82 ± 0.40 | 0.00297 | **0.00/ep** ← collapse | MLP |
| G4 | Recovering Cooperator | −1.62 ± 0.39 | 0.00298 | **0.00/ep** ← persists | MLP + Attention |
| G5 | **Fire Discipline** | −2.07 ± 0.38 | **0.00373** | **0.33/ep** ← GRU helps | Attn + GRU |

> **V2:** All games trained with `--death_penalty 2.0`. Mean returns are lower (penalty subtracts). Compare raw kill density across games.  
> **G5 details (stochastic):** preservation +0.675, ep length 181 steps, 2,007,048 training steps, 825,481 parameters

---

## Constraint Stack

| Level | Death Penalty | Ammo Limit | Stamina Decay | 60/40 Team Reward | Gaussian Fog |
|-------|:------------:|-----------|---------------|-------------------|-------------|
| G1 | ✅ 2.0 | — | — | — | — |
| G2 | ✅ 2.0 | ✅ 15 arrows/ep | — | — | — |
| G3 | ✅ 2.0 | ✅ | ✅ move −0.01, attack −0.05 | — | — |
| G4 | ✅ 2.0 | ✅ | ✅ | ✅ | — |
| G5 | ✅ 2.0 | ✅ | ✅ | ✅ | ✅ σ=0.3 |

---

## Architecture Evolution

```
G1–G3:  obs(flat) ──► Linear(obs,256)─ReLU─Linear(256,256)─ReLU ──► policy / value heads
                       [MLP backbone, ~200K params]

G4:     obs(27×5) ──► EntityAttentionEncoder (4-head self-attn, masked pooling)
                    ──► Linear(256,256)─ReLU ──► policy / value heads
                       [MLP + Attention, ~467K params]

G5:     obs(27×5) ──► EntityAttentionEncoder
                    ──► GRUCell(256,256) [h_t = f(z_t, h_{t-1})]
                    ──► Linear(256,256)─ReLU ──► policy / value heads
                       [Attn + GRU, 825,481 params — transfer-loaded from G4]
```

- **Algorithm:** MAPPO — all agents share one network; each observes locally
- **Reward:** `R_i = 0.6×R_self + 0.4×R_team` from G4 onward; pure kill reward for G1–G3; death penalty (−2.0) on all games
- **Logging:** TensorBoard reward decomposition (aggression vs. preservation) per step
- **Device:** Auto-detected — CUDA (NVIDIA) → MPS (Apple Silicon) → CPU

---

## Project Structure

```
ADLProject2/
├── src/
│   ├── models/
│   │   └── mappo_net.py     # MAPPONet: MLP / Attention / Attention+GRU
│   ├── wrappers/
│   │   └── kaz_wrapper.py   # KAZWrapper — toggles ammo, stamina, team reward, fog
│   ├── train.py             # MAPPO training loop (PPO + GAE, TensorBoard logging)
│   ├── evaluate.py          # Checkpoint evaluation, metrics JSON, video recording
│   ├── phase5.py            # Phase 5 analysis — all 4 evaluation artifacts
│   ├── utils.py             # Device auto-detection (CUDA / MPS / CPU)
│   └── test_env.py          # Environment smoke test
├── models/                  # Checkpoints — game1/final.pt … game5/final.pt
├── results/
│   ├── kill_density_evolution.png   # G1→G5 ablation bar chart
│   ├── collapse_graph.png           # Stress test: spawn pressure vs. kill density
│   ├── saliency_comparison.png      # G1 gradient saliency vs G5 attention heatmap
│   ├── demo_sidebyside.mp4          # 30s side-by-side: Greedy Soldier vs Fire Discipline
│   ├── final_ablation_table.md      # Full G1–G5 metrics table
│   ├── phase4_observations.md       # G5 training analysis and findings
│   ├── game*_eval_results*.json     # Per-game evaluation JSONs (stoch + det)
│   └── tensorboard/                 # TensorBoard reward decomposition logs
├── docs/
│   ├── report.pdf                   # 8-page academic paper (LaTeX compiled)
│   ├── report.tex                   # LaTeX source
│   ├── references.bib               # Bibliography (10 citations)
│   ├── presentation_guide.md        # 13-slide 15-min talk blueprint
│   └── phase*_plan.md               # Per-phase implementation plans
├── requirements.txt
├── plan.md                  # Full project roadmap (all phases ✅)
└── rules.md                 # Execution rules
```

---

## Platform Compatibility

The codebase auto-detects the best available device at runtime — no code changes required.

| Platform | Device Used | Notes |
|----------|-------------|-------|
| Linux / NVIDIA GPU | **CUDA** | Recommended. Trained on RTX 5070 Ti (~60 min for G5) |
| macOS (Apple Silicon) | **MPS** | Metal Performance Shaders. All ops supported. |
| macOS (Intel) / CPU-only | **CPU** | Fully functional, significantly slower (~10–20× vs. GPU) |
| Windows | **WSL2 required** | Run all commands inside Ubuntu WSL2 terminal |

> **Windows users:** Install WSL2 with `wsl --install` in PowerShell (Admin), then follow Linux setup steps inside the Ubuntu terminal.

---

## Setup

### Prerequisites
- Python 3.10–3.12
- For CUDA: NVIDIA GPU with CUDA 12.x drivers

```bash
# 1. Clone
git clone https://github.com/OneTrickBreach/ADL-Final-Project.git ADLProject2
cd ADLProject2

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows WSL2 (same as Linux)
```

### Install PyTorch (choose your platform)

```bash
# NVIDIA GPU (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Apple Silicon — MPS (macOS 12.3+)
pip install torch torchvision

# CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Install remaining dependencies

```bash
pip install "pettingzoo[butterfly]" gymnasium supersuit tensorboard opencv-python matplotlib
```

### Verify installation

```bash
./.venv/bin/python src/test_env.py
# Expected output: "Smoke test PASSED"
```

---

## Quick Start

```bash
# Train Game 1 — baseline MLP, death penalty (~1M steps)
./.venv/bin/python src/train.py --game_level 1 --total_timesteps 1000000 --death_penalty 2.0

# Train Game 5 — Attention+GRU, full constraints, transfer from G4 (~2M steps)
./.venv/bin/python src/train.py \
    --game_level 5 --arch attention_gru \
    --transfer_from models/game4/final.pt \
    --total_timesteps 2000000 --max_cycles 900 --death_penalty 2.0

# Evaluate stochastic policy (10 episodes)
./.venv/bin/python src/evaluate.py --checkpoint models/game5/final.pt --episodes 10

# Evaluate deterministic (argmax) policy
./.venv/bin/python src/evaluate.py --checkpoint models/game5/final.pt --episodes 10 --deterministic

# Record gameplay videos → results/game5_demo/
./.venv/bin/python src/evaluate.py --checkpoint models/game5/final.pt --episodes 3 --record

# Generate all Phase 5 artifacts (plots + side-by-side video)
./.venv/bin/python src/phase5.py

# Launch TensorBoard
tensorboard --logdir results/tensorboard
```

---

## Full Training Sequence (G1 → G5)

```bash
# Game 1 — MLP, kill reward + death penalty
./.venv/bin/python src/train.py --game_level 1 --total_timesteps 1000000 --death_penalty 2.0

# Game 2 — MLP + ammo limit + death penalty
./.venv/bin/python src/train.py --game_level 2 --total_timesteps 1000000 --death_penalty 2.0

# Game 3 — MLP + ammo + stamina + death penalty
./.venv/bin/python src/train.py --game_level 3 --total_timesteps 1000000 --death_penalty 2.0

# Game 4 — Entity Attention + team reward + death penalty
./.venv/bin/python src/train.py --game_level 4 --arch attention --total_timesteps 1500000 --death_penalty 2.0

# Game 5 — Attention+GRU + fog + death penalty (transfer from G4)
./.venv/bin/python src/train.py \
    --game_level 5 --arch attention_gru \
    --transfer_from models/game4/final.pt \
    --total_timesteps 2000000 --max_cycles 900 --death_penalty 2.0
```

---

## Phase 5 Artifacts

All artifacts are pre-generated in `results/`:

| File | Description |
|------|-------------|
| `kill_density_evolution.png` | G1→G5 kill density + preservation bar chart |
| `collapse_graph.png` | Spawn-pressure stress test — G1 vs G5 at 0.7×/1×/1.4×/2× load |
| `saliency_comparison.png` | G1 MLP gradient saliency vs G5 attention weight heatmap (27×27) |
| `demo_sidebyside.mp4` | 30-second side-by-side: Greedy Soldier vs Fire Discipline |

To regenerate:
```bash
./.venv/bin/python src/phase5.py
```

---

## Documentation

| Document | Location | Description |
|----------|----------|-------------|
| Academic paper | `docs/report.pdf` | 9-page LaTeX paper — methods, results, explainability |
| LaTeX source | `docs/report.tex` + `docs/references.bib` | |
| 15-min talk guide | `docs/presentation_guide.md` | 13 slides with speaking notes and timing |
| Final ablation table | `results/final_ablation_table.md` | G1–G5 full metrics |
| G5 training analysis | `results/phase4_observations.md` | Deterministic passivity diagnosis |

---

## Progress

| Phase | Status | Key Result |
|-------|--------|------------|
| **Phase 1** — G1 Villain Baseline | ✅ | Kill density 0.01051 (V2), mean return 3.98 ± 3.80 (death penalty active) |
| **Phase 2** — G2/G3 Resource Scarcity | ✅ | G2: 0.00308 · G3: 0.00297 · **death penalty does NOT fix passivity** |
| **Phase 3** — G4 Recovering Cooperator | ✅ | Density 0.00298, preservation 0.575 · attention alone still insufficient |
| **Phase 4** — G5 Fire Discipline | ✅ | Raw kill density **0.00373** (+25% vs G4) · GRU + death penalty = best combo |
| **Phase 5** — Evaluation & Explainability | ✅ | 4 artifacts regenerated · V2 report + presentation updated |
| **V2** — Death Penalty Iteration | ✅ | `--death_penalty 2.0` all games · confirms GRU is necessary, penalty is complementary |
