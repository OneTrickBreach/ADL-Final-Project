## 🛡️ Project: The Evolution of Fire Discipline (5-Game Ablation)

### Core Model Architecture
* **Type:** Multi-Agent PPO (MAPPO) with Shared-Parameter Backbone.
* **Modification:** Multi-Objective Reward Heads with a **60/40 Weighting** for teammate health.
* **Hardware:** RTX 5070 Ti (optimized via `torch.cuda`).

---

### Project Directory Layout
```
ADLProject2/
├── src/                        # All source code (wrappers, models, training loops)
│   ├── wrappers/               # KAZWrapper and game-level modifier classes
│   ├── models/                 # MAPPO network definitions and reward heads
│   └── train.py                # Main training entry point
├── notebooks/                  # Exploratory analysis and saliency map prototyping
├── data/
│   ├── raw/                    # Unprocessed environment recordings / replays
│   └── processed/              # Preprocessed observations for offline analysis
├── models/                     # Saved checkpoints per Game level (G1–G5)
├── results/                    # Output plots, ablation tables, metrics, logs
│   └── tensorboard/            # SummaryWriter logs (Reward Decomposition per run)
├── docs/                       # Documentation and writeup assets
├── plan.md                     # This file — check before every task
├── rules.md                    # Execution rules for this project
└── .venv/                      # Project-scoped virtual environment
```

**Artifact mapping by phase:**
* **Phase 1–4 training scripts** → `src/train.py`, wrappers → `src/wrappers/`
* **Checkpoints (G1–G5)** → `models/game{1..5}/`
* **TensorBoard logs** → `results/tensorboard/`
* **Saliency maps & video demos** → `results/` and `notebooks/`
* **Ablation table & collapse graph** → `results/`

---

### Phase 1: The Villain Baseline (Game 1) ✅
* **Setup:** Default PettingZoo KAZ (Infinite ammo, no stamina costs, perfect vision).
* **Training:** Standard $R = \text{Kills}$. Trained 502,586 steps / 409 episodes.
* **Behavior:** The **"Greedy Soldier."** Charging zombies blindly, overlapping paths, zero coordination.
* **Results:** Mean return 5.80 ± 2.94 | Raw kill density 0.01089 | Mean ep length 533 steps.
* **Artifacts:** `models/game1/final.pt`, `results/game1_baseline_metrics.json`, `results/game1_demo/`.

### Phase 2: Resource Scarcity (Games 2 & 3) ✅
* **Game 2 (+ Ammo Restriction):**
    * **Wrapper:** Archer has 15 arrows/episode, −0.5 dry-fire penalty.
    * **Training:** 504,134 steps / 676 episodes.
    * **Behavior:** The **"Risk Avoider."** Archer suppresses shooting entirely (0.15% shoot rate under deterministic eval); raw kill density drops 2.5× from G1.
    * **Results:** Mean return 0.30 ± 0.39 | Raw kills 0.85 | Raw kill density 0.00431 | Mean ep length 197 steps.
    * **Artifacts:** `models/game2/final.pt`, `results/game2_eval_results.json`, `results/game2_demo/`.
* **Game 3 (+ Stamina Decay):**
    * **Wrapper:** Knight pays 0.01/move + 0.05/attack on top of G2 ammo limits.
    * **Training:** 500,641 steps / 697 episodes.
    * **Behavior:** The **"Fully Passive."** Knight drops attack to 0% under deterministic eval; penalty avoidance dominates the reward signal.
    * **Results:** Mean return 0.15 ± 0.24 | Raw kills 0.50 | Raw kill density 0.00279 | Mean ep length 179 steps.
* **Key Finding:** Penalties too strong relative to kill reward → agents learn **avoidance, not efficiency**. Phase 3 should tune penalty magnitudes before stacking 60/40 comrade reward.
    * **Artifacts:** `models/game3/final.pt`, `results/game3_eval_results.json`, `results/game3_demo/`.
* **Analysis:** `results/phase2_ablation_table.md`, `results/phase2_observations.md`.

### Phase 3: The Altruistic Hero (Game 4) ✅
* **Game 4 (+ 60/40 Comrade Healthcare + Entity Self-Attention):**
    * **Logic:** Reward for Agent $i$: $R_i = (0.6 \times R_{\text{self}}) + (0.4 \times R_{\text{team}})$.
    * **Architecture:** EntityAttentionEncoder — multi-head self-attention over (27 entities × 5 features) observation structure. 430K params (vs 200K MLP).
    * **Training:** 507,960 steps / 706 episodes.
    * **Behavior:** The **"Recovering Cooperator."** Kill density reverses downward trend (+29% from G3). Team reward flows (mean preservation +0.68). Archers shoot 8.6% stochastically (up from 0% in G3). But deterministic policy remains passive (0 kills) — the policy mode hasn't shifted to offensive.
    * **Results:** Mean return 0.53 ± 0.35 | Raw kills 0.68 | Raw kill density 0.00361 | Preservation 0.68 | Mean ep length 187 steps.
    * **Artifacts:** `models/game4/final.pt`, `results/game4_eval_results.json`, `results/game4_demo/`.
* **Analysis:** `results/ablation_table.md`, `results/phase3_observations.md`.

### Phase 4: Tactical Uncertainty (Game 5 - The Final Hero)
* **Game 5 (+ Gaussian Fog):**
    * **Wrapper:** Apply a Gaussian blur/noise to observations.
    * **The "Fire Discipline" Peak:** Agents must integrate resource costs (G2/G3) and teammate safety (G4) with visual uncertainty.
    * **Goal:** Explicit coordination and perimeter defense.

---

### Phase 5: Evaluation & Explainability
1.  **Quantitative:**
    * **Ablation Table:** A table comparing Kills/Resource Ratio across all 5 Games.
    * **The "Collapse" Graph:** Show how Game 1 fails at 1.5x zombie density while Game 5 survives via efficiency.
2.  **Qualitative (The "Explain" Box):**
    * **Saliency Maps:** Show Game 1 attending only to the "Nearest Enemy" vs. Game 5 attending to the "Teammate + Fog Border."
    * **Video Demo:** Side-by-side of the "Greedy Soldier" vs. the "Tactical Unit."

---

### 🚀 Immediate Next Steps:
1.  ~~**Code the Wrappers:** Create a single `KAZWrapper` class that can toggle Ammo, Stamina, and Fog based on a `game_level` argument.~~ ✅ Done
2.  ~~**Reward Scalarizer:** Implement the $0.6/0.4$ logic in your environment's `step()` function so it's baked into the reward signal before it hits the PPO agent.~~ ✅ Done
3.  ~~**Phase 2:** Train Game 2 (ammo restriction) and Game 3 (stamina decay) — wrappers are already wired in `KAZWrapper`.~~ ✅ Done
4.  ~~**Phase 3:** Train Game 4 (60/40 comrade healthcare + entity self-attention).~~ ✅ Done
5.  **Phase 4:** Train Game 5 (Gaussian fog + GRU recurrence) — the Final Hero.