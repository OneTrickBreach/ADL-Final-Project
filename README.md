# The Evolution of Fire Discipline
### ADL CS7180 Final Project — 5-Game Ablation Study

**Team:** Ishan Biswas, Elizabeth Coquillette, Nishant Suresh

---

## Overview

A multi-agent reinforcement learning study using **MAPPO (Multi-Agent PPO with Shared-Parameter Backbone)** on the PettingZoo **Knights, Archers & Zombies (KAZ)** environment.

The project runs a 5-game ablation that incrementally adds constraints to observe the emergence of coordinated, disciplined agent behavior — from a "Greedy Soldier" baseline to a fully tactical unit operating under resource limits, teammate awareness, and perceptual uncertainty.

| Game | Constraint Added | Emergent Behavior |
|------|-----------------|-------------------|
| G1 | None (baseline) | Greedy Soldier |
| G2 | Ammo restriction | Trigger Discipline |
| G3 | Stamina decay | Economic Positioning |
| G4 | 60/40 teammate reward sharing | Guardian / Coordinator |
| G5 | Gaussian fog + GRU recurrence | **Fire Discipline** (69.9% det. attack rate) |

---

## Architecture

- **Algorithm:** MAPPO — shared-parameter policy network across agents
- **Reward:** Multi-objective heads; $R_i = (0.6 \times R_{\text{self}}) + (0.4 \times R_{\text{team}})$ from Game 4 onward
- **Logging:** TensorBoard `SummaryWriter` with Reward Decomposition (Aggression vs. Preservation)
- **Hardware:** NVIDIA RTX 5070 Ti — `torch.device("cuda")` throughout

---

## Project Structure

```
ADLProject2/
├── src/
│   ├── wrappers/        # KAZWrapper — toggles ammo, stamina, fog by game_level
│   ├── models/          # MAPPO network definitions and reward heads
│   ├── train.py         # Main training entry point
│   ├── evaluate.py      # Checkpoint evaluation, metrics export, video recording
│   └── test_env.py      # Environment smoke test
├── notebooks/           # Saliency map prototyping, exploratory analysis
├── data/
│   ├── raw/             # Unprocessed environment recordings
│   └── processed/       # Preprocessed observations
├── models/              # Checkpoints per game level (game1/ … game5/)
├── results/
│   ├── tensorboard/             # SummaryWriter logs (Reward Decomposition per run)
│   ├── game1_demo/ … game5_demo/ # Recorded gameplay videos per game
│   ├── final_ablation_table.md  # Full G1–G5 ablation metrics
│   └── phase4_observations.md   # G5 training analysis and findings
├── docs/                # Phase plans and documentation
├── requirements.txt     # Pinned dependencies
├── plan.md              # Project roadmap — check before every task
└── rules.md             # Execution rules for this project
```

---

## Platform Compatibility

> **Windows users:** This project does **not** run natively on Windows. You must use **WSL2** (Windows Subsystem for Linux) with an Ubuntu distro (22.04 or 24.04 recommended).
> - Install WSL2: `wsl --install` in PowerShell (Admin)
> - Then follow all setup steps inside the WSL/Ubuntu terminal.
>
> **macOS / Linux:** Works natively — no extra steps needed.

---

## Setup

> Requires Python 3.12 and an NVIDIA GPU with CUDA 12.8+ drivers.

```bash
# 1. Clone the repo
git clone https://github.com/OneTrickBreach/ADL-Final-Project.git ADLProject2
cd ADLProject2

# 2. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "pettingzoo[butterfly]" gymnasium supersuit tensorboard opencv-python matplotlib "ray[rllib]"
```

---

## Quick Start

```bash
# Smoke test — verify env + wrapper
./.venv/bin/python src/test_env.py

# Train Game 1 baseline (500k steps, ~20 min on RTX 5070 Ti)
./.venv/bin/python src/train.py --game_level 1 --total_timesteps 500000

# Train Game 5 — attention_gru + fog, transfer weights from G4 (~60 min)
./.venv/bin/python src/train.py \
    --game_level 5 --arch attention_gru \
    --transfer_from models/game4/final.pt \
    --total_timesteps 1500000 --max_cycles 900

# Evaluate a checkpoint (stochastic, 10 episodes)
./.venv/bin/python src/evaluate.py --checkpoint models/game5/final.pt --episodes 10

# Evaluate deterministic policy
./.venv/bin/python src/evaluate.py --checkpoint models/game5/final.pt --episodes 10 --deterministic
```

---

## Watch & Record Gameplay

```bash
# Watch Game 1 live in a window
./.venv/bin/python src/evaluate.py --checkpoint models/game1/final.pt --episodes 3 --render

# Record episodes as mp4 videos → results/game1_demo/
./.venv/bin/python src/evaluate.py --checkpoint models/game1/final.pt --episodes 3 --record

# Play a recorded video (install vlc/mpv if needed)
vlc results/game1_demo/episode_1.mp4
```

---

## Progress

| Phase | Status | Key Result |
|-------|--------|------------|
| **Phase 1** — Villain Baseline (G1) | ✅ Complete | Raw kill density 0.01089, mean return 5.80 ± 2.94 |
| **Phase 2** — Resource Scarcity (G2, G3) | ✅ Complete | G2: raw kills 0.85, density 0.00431 · G3: raw kills 0.50, density 0.00279 · **Finding: penalty avoidance, not efficiency** |
| **Phase 3** — Altruistic Hero (G4) | ✅ Complete | Raw kills 0.68, density 0.00361 (+29% from G3), preservation +0.68 · **Attention + team reward reverses kill decline** |
| **Phase 4** — Tactical Uncertainty (G5) | ✅ Complete | Det. attack rate **69.9%** (vs ~0% G4), kill density +0.00165 (+27% shaped vs G4), pres +0.525 · **GRU resolves deterministic passivity** |
| **Phase 5** — Evaluation & Explainability | ✅ Complete | Full ablation table `results/final_ablation_table.md` · Phase 4 analysis `results/phase4_observations.md` · Demo videos `results/game5_demo/` |

---

## Evaluation Plan

- **Ablation Table:** Kills/Resource Ratio across all 5 games
- **Collapse Graph:** Game 1 failing at 1.5× zombie density vs. Game 5 surviving via efficiency
- **Saliency Maps:** Attention shift from "Nearest Enemy" (G1) to "Teammate + Fog Border" (G5)
- **Video Demo:** Side-by-side "Greedy Soldier" vs. "Tactical Unit"
