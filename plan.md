# Project Plan: Multi-Agent Tactical Divergence in KAZ

## Phase 0: Project Initialization & Git Setup
- [ ] Ensure IDE is connected to `WSL: Ubuntu` and opened in `~/adl_final_kaz`.
- [ ] Initialize Git repository: `git init`.
- [ ] Create a `.gitignore` to exclude `.venv/`, `results/`, and `__pycache__/`.
- [ ] Link to GitHub: `git remote add origin <your-repo-url>`.
- [ ] Perform first commit: `git add. && git commit -m "Initial structure"`.

## Phase 1: High-Performance Environment Setup
- [ ] Create virtual environment: `python3 -m venv.venv`.
- [ ] Activate environment and install **Torch with CUDA 12.4 support** (Optimized for 5070 Ti):
  - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
- [ ] Install MARL stack:
  - `pip install 'pettingzoo[butterfly]' supersuit 'ray[rllib]' tensorboard`
- [ ] Verify GPU visibility: `python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()} - Device: {torch.cuda.get_device_name(0)}')"`

## Phase 2: Baseline "Selfish" Agent Logic
- [ ] Implement `src/env_setup.py`:
  - Initialize `knights_archers_zombies_v10`.
  - Apply SuperSuit wrappers: `color_reduction_v0`, `resize_v1` (84x84), `frame_stack_v1` (3), and `black_death_v3`.
- [ ] Implement `src/train.py`:
  - Set up a standard PPO configuration using RLlib's Multi-Agent API.
  - Run a 100-iteration baseline training session.
  - Store results in `results/baseline/`.

## Phase 3: The "Novelty" Layer - Tactical Divergence
- [ ] Create a custom Reward Wrapper in `src/env_setup.py` to decompose signals:
  - **Knight Reward:** $+1$ for kills, $-0.5$ if Archer dies (Preservation bias).
  - **Archer Reward:** $+1$ for kills, $-2.0$ if Zombies reach the border (Border integrity bias).
- [ ] Update `src/train.py` to use these divergent rewards.
- [ ] Implement a **Hyperparameter Sweep** over the "Preservation weight" to find the Pareto Front.

## Phase 4: Qualitative Evaluation & "The Story"
- [ ] Implement `src/visualize.py`:
  - Generate 30-second GIFs of the trained agents.
  - Highlight "Covering Fire" behaviors (where Archer protects a reloading Knight).
- [ ] Export TensorBoard logs to analyze the "Selfish Collapse" vs. "Tactical Coordination" curves.

## Phase 5: Final Submission Readiness
- [ ] Finalize `README.md` with instructions on how to reproduce the training.
- [ ] Push all code to GitHub: `git push origin main`.