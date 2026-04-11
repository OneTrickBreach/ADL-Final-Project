# Phase 2 — Ablation Comparison Table

## Evaluation Results (10 episodes, stochastic policy)

| Metric | Game 1 (Baseline) | Game 2 (+ Ammo) | Game 3 (+ Stamina) |
|--------|-------------------|------------------|---------------------|
| **Mean Return** | 5.80 ± 2.94 | 0.30 ± 0.39 | 0.15 ± 0.24 |
| **Mean Raw Kills** | 5.80 | 0.85 | 0.50 |
| **Mean Aggression** (shaped) | 5.80 | 0.30 | 0.15 |
| **Preservation** | 0.00 | 0.00 | 0.00 |
| **Mean Ep Length** | 532.7 | 197.0 | 179.0 |
| **Raw Kill Density** | 0.01089 | 0.00431 | 0.00279 |
| **Shaped Kill Density** | 0.01089 | 0.00152 | 0.00085 |

### Penalty Drag (Raw Kills − Shaped Aggression)

| | Game 1 | Game 2 | Game 3 |
|---|--------|--------|--------|
| **Penalty drag per episode** | 0.00 | −0.55 | −0.35 |
| **% of raw kills lost to penalties** | 0% | 65% | 70% |

> G2 penalty drag is from dry-fire penalties (−0.5 each).
> G3 penalty drag includes both dry-fire and stamina costs (0.01/move + 0.05/attack).

## Training Summary

| | Game 1 | Game 2 | Game 3 |
|---|--------|--------|--------|
| **Total Timesteps** | 502,586 | 504,134 | 500,641 |
| **Episodes** | 409 | 676 | 697 |
| **Mean Train Return (last 50)** | 4.57 | 0.66 | 0.20 |
| **Mean Train Ep Length** | 436.9 | 199.0 | 182.2 |

## Deterministic Policy Analysis

When evaluated with `--deterministic` (argmax action selection):

| | Game 1 | Game 2 | Game 3 |
|---|--------|--------|--------|
| **Mean Return** | 1.23 ± 0.91 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Shoot/Attack Action %** | 39.9% | 0.15% | 0.0% |
| **Dominant Action** | Shoot (4) | Move fwd (1) | Turn (5) |

> The deterministic policy reveals that G2/G3 learned to **avoid offensive actions entirely**.
> Stochastic evaluation masks this by generating accidental kills through random sampling.
> This is **risk avoidance**, not strategic efficiency — a key finding for the ablation narrative.

## Key Observations

1. **Raw kill density drops 2.5× from G1→G2 and 3.9× from G1→G3** — these are clean counts uncontaminated by penalties.
2. **Shaped kill density overstates the drop** (7.2× G1→G2, 12.8× G1→G3) because it includes penalty costs in the numerator.
3. **Episode length collapses** from 533→197→179 steps, meaning agents die much faster under constraints — they aren't learning to survive longer, they're dying sooner because they stop fighting back.
4. **G2/G3 train for more episodes** in the same timestep budget (409→676→697), confirming shorter survival times.
5. **Policy collapse under deterministic eval** proves that the penalties dominated the reward signal, teaching avoidance instead of efficiency.
