# Phase 2 Ablation Table — G1 vs G2 vs G3

**Evaluation:** 10 episodes per game, seed=123, checkpoints trained to ~500k steps each.

| Metric | Game 1 (Baseline) | Game 2 (+ Ammo) | Game 3 (+ Ammo + Stamina) |
|--------|-------------------|-----------------|--------------------------|
| **Mean return** | 5.55 ± 3.74 | 0.55 ± 0.28 | 0.10 ± 0.32 |
| **Mean aggression** | 5.55 | 0.55 | 0.10 |
| **Mean preservation** | 0.00 | 0.00 | 0.00 |
| **Mean ep length** | 534.4 | 197.0 | 201.0 |
| **Kill density** | 0.01039 | 0.00279 | 0.00049 |
| **Steps trained** | 501,838 | 507,273 | 500,839 |
| **Episodes trained** | 401 | 642 | 663 |

## Key Observations

- **Return drop:** G1 → G2 sees a ~90% drop in mean return (5.55 → 0.55). G2 → G3 drops another ~82% (0.55 → 0.10). Resource costs directly suppress raw reward accumulation.
- **Kill density:** Falls by ~3.7× from G1 to G2 and another ~5.7× from G2 to G3. Agents are killing far less frequently per timestep.
- **Episode length:** G1 episodes run much longer (534 steps avg) because unconstrained agents survive and keep fighting. G2/G3 episodes are shorter (~200 steps), suggesting agents die faster or the environment terminates sooner under pressure.
- **Preservation = 0:** Expected — team reward sharing is disabled until Game 4.
- **Entropy (training logs):** G3 entropy collapsed significantly during training (1.79 → 0.83), indicating the policy became much more deterministic. This is consistent with "Economic Positioning" — the knight converges on a narrow set of cost-efficient actions.
- **G2 entropy** remained moderate (~1.50), suggesting the archer still explores but is more selective about shooting.
