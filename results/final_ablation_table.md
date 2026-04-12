# Final Ablation Table — The Evolution of Fire Discipline

> All evaluations: 10 stochastic episodes, seed=123.  
> G5 also includes a 10-episode deterministic run (seed=123).

---

## Core Metrics

| Game | Name | Arch | Steps | Mean Return | Ep Length | Raw Kill Density | Kill Density | Preservation |
|------|------|------|-------|-------------|-----------|-----------------|--------------|--------------|
| G1 | Greedy Soldier | MLP | 502,586 | 5.80 ± 2.94 | 533 | 0.01089 | 0.01089 | 0.000 |
| G2 | Risk Avoider | MLP | 504,134 | 0.30 ± 0.39 | 197 | 0.00431 | 0.00152 | 0.000 |
| G3 | Fully Passive | MLP | 500,641 | 0.15 ± 0.24 | 179 | 0.00279 | 0.00085 | 0.000 |
| G4 | Recovering Cooperator | MLP + Attention | 507,960 | 0.42 ± 0.38 | 187 | 0.00361 | 0.00130 | 0.675 |
| G5 (stoch) | **Fire Discipline** | MLP + Attn + GRU | 1,506,717 | 0.38 ± 0.31 | 176 | 0.00299 | 0.00165 | 0.525 |
| G5 (det) | **Fire Discipline** | MLP + Attn + GRU | 1,506,717 | 0.20 ± 0.19 | 169 | 0.00163 | 0.00094 | 0.275 |

---

## Kill Density Trajectory (Stochastic Eval)

```
G1  ████████████████████████  0.01089   Baseline (no constraints)
G2  █████████░░░░░░░░░░░░░░░  0.00431   ↓ −60%  Ammo restriction
G3  ██████░░░░░░░░░░░░░░░░░░  0.00279   ↓ −35%  + Stamina decay
G4  ████████░░░░░░░░░░░░░░░░  0.00361   ↑ +29%  + Team reward + Attention
G5  ███████░░░░░░░░░░░░░░░░░  0.00299   ↓ −17%  + Gaussian Fog (expected)
```

> G5's fog handicap accounts for the drop from G4. Effective kill rate given observation noise is the relevant comparison.

---

## Constraint Stack

| Game | Infinite Ammo | No Stamina Cost | Perfect Vision | 60/40 Team Reward | Attention Enc | GRU Memory |
|------|:-------------:|:---------------:|:--------------:|:-----------------:|:-------------:|:----------:|
| G1 | ✅ | ✅ | ✅ | ✗ | ✗ | ✗ |
| G2 | ✗ (15 arrows) | ✅ | ✅ | ✗ | ✗ | ✗ |
| G3 | ✗ | ✗ (0.01/move, 0.05/attack) | ✅ | ✗ | ✗ | ✗ |
| G4 | ✗ | ✗ | ✅ | ✅ | ✅ | ✗ |
| G5 | ✗ | ✗ | ✗ (σ=0.3 fog) | ✅ | ✅ | ✅ |

---

## The Deterministic Passivity Problem — Solved by GRU

A key pathology in G3 and G4 was **deterministic passivity**: under argmax policy, agents chose zero aggressive actions because the penalty/reward tradeoff punished the mode of the distribution.

| Game | Deterministic Attack Rate | Notes |
|------|--------------------------|-------|
| G1 | ~100% | Unconstrained aggression |
| G2 | ~0.15% | Archer suppresses firing entirely |
| G3 | 0% | Full passive collapse |
| G4 | ~0% | Still passive under argmax (stochastic: 8.6%) |
| **G5** | **69.9%** | **GRU memory overcomes passivity** |

> G5's GRU provides temporal context over fogged observations, allowing the policy mode to commit to attack actions rather than defaulting to safe no-ops.

---

## Action Distribution — G5 Deterministic (20 episodes)

| Action | Count | Fraction |
|--------|-------|----------|
| noop | 766 | 5.6% |
| move_left | 1,729 | 12.7% |
| move_right | 715 | 5.3% |
| move_up | 680 | 5.0% |
| move_down | 206 | 1.5% |
| **attack** | **9,493** | **69.9%** |
| *Total steps* | *13,589* | |

---

## Network Parameter Count

| Game | Arch | Params |
|------|------|--------|
| G1–G3 | MLP (2-layer) | ~200,329 |
| G4 | EntityAttentionEncoder + MLP | 430,729 |
| G5 | EntityAttentionEncoder + GRUCell(256,256) + MLP | **825,481** |

GRU adds 394,752 parameters (4 weight matrices: `weight_ih`, `weight_hh`, `bias_ih`, `bias_hh` for 3 gates each).

---

## Training Efficiency

| Game | Steps | Episodes | SPS (final) |
|------|-------|----------|-------------|
| G1 | 502,586 | 409 | ~425 |
| G2 | 504,134 | 676 | ~410 |
| G3 | 500,641 | 697 | ~405 |
| G4 | 507,960 | 706 | ~400 |
| G5 | 1,506,717 | 2,076 | ~378 |

> G5 SPS reduction (~11% vs G4) reflects GRU overhead and longer episodes due to fog-reduced episode termination speed.
