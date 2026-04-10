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
| G5 | Gaussian fog on observations | Fire Discipline Peak |

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
│   └── train.py         # Main training entry point
├── test/
│   ├── smoke_test.py    # Full component check (imports, env, CUDA, TensorBoard)
│   └── test_render.py   # GUI render test
├── notebooks/           # Saliency map prototyping, exploratory analysis
├── data/
│   ├── raw/             # Unprocessed environment recordings
│   └── processed/       # Preprocessed observations
├── models/              # Checkpoints per game level (game1/ … game5/)
├── results/
│   └── tensorboard/     # SummaryWriter logs
├── docs/                # Documentation and writeup assets
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

## Running Tests

```bash
# Full component smoke test (imports, env rollout, CUDA, TensorBoard)
.venv/bin/python test/smoke_test.py

# GUI render test — opens the KAZ game window with random agents
.venv/bin/python test/test_render.py
```

---

## Evaluation Plan

- **Ablation Table:** Kills/Resource Ratio across all 5 games
- **Collapse Graph:** Game 1 failing at 1.5× zombie density vs. Game 5 surviving via efficiency
- **Saliency Maps:** Attention shift from "Nearest Enemy" (G1) to "Teammate + Fog Border" (G5)
- **Video Demo:** Side-by-side "Greedy Soldier" vs. "Tactical Unit"
