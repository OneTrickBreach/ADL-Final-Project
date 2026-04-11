# Ablation Comparison Table (G1–G4)

## Evaluation Results (10 episodes, stochastic policy)

| Metric | G1 (Baseline) | G2 (+ Ammo) | G3 (+ Stamina) | G4 (+ Team + Attention) |
|--------|---------------|-------------|----------------|-------------------------|
| **Mean Return** | 5.80 ± 2.94 | 0.30 ± 0.39 | 0.15 ± 0.24 | 0.53 ± 0.35 |
| **Mean Raw Kills** | 5.80 | 0.85 | 0.50 | 0.68 |
| **Mean Aggression** (shaped) | 5.80 | 0.30 | 0.15 | 0.43 |
| **Preservation Reward** | 0.00 | 0.00 | 0.00 | 0.68 |
| **Mean Ep Length** | 532.7 | 197.0 | 179.0 | 187.0 |
| **Raw Kill Density** | 0.01089 | 0.00431 | 0.00279 | 0.00361 |
| **Shaped Kill Density** | 0.01089 | 0.00152 | 0.00085 | 0.00231 |
| **Architecture** | MLP | MLP | MLP | Attention |

### Kill Density Trend (north-star metric)

```
G1: 0.01089  ████████████████████████████████  (baseline)
G2: 0.00431  ████████████▋                     (−60%)
G3: 0.00279  ████████▏                         (−74%)
G4: 0.00361  ██████████▍                       (+29% from G3, reversal!)
```

**G4 reverses the downward trend.** Raw kill density increased 29% from G3 despite carrying all prior constraints.

## Training Summary

| | G1 | G2 | G3 | G4 |
|---|-----|-----|-----|-----|
| **Total Timesteps** | 502,586 | 504,134 | 500,641 | 507,960 |
| **Episodes** | 409 | 676 | 697 | 706 |
| **Mean Train Return (last 50)** | 4.57 | 0.66 | 0.20 | 0.46 |
| **Mean Train Ep Length** | 436.9 | 199.0 | 182.2 | 188.2 |
| **Architecture** | MLP (200K) | MLP (200K) | MLP (200K) | Attention (431K) |

## Deterministic Policy Analysis

| | G1 | G2 | G3 | G4 |
|---|-----|-----|-----|-----|
| **Mean Return** | 1.23 ± 0.91 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| **Knight Attack %** | — | — | 0.0% | 0.0% |
| **Archer Shoot %** | 39.9% | 0.15% | 0.0% | 0.0% |
| **Knight Dominant** | — | — | Turn (5) | Turn (5) |
| **Archer Dominant** | Shoot (4) | Move fwd (1) | Turn (5) | Move fwd (1) |

### Stochastic Action Distribution (G4)

| | Knight | Archer |
|---|--------|--------|
| **Attack/Shoot %** | 0.0% | 8.6% |
| **Move %** | 3.5% | 62.2% |
| **Turn %** | 96.1% | 23.8% |
| **Noop %** | 0.4% | 5.4% |

> **Honest finding:** The deterministic policy is still passive (0 kills), same as G2/G3.
> But the stochastic policy shows meaningful improvement — archer shoot rate jumped from 0% (G3) to 8.6%.
> Kills come from stochastic sampling, not intentional argmax selection. Team reward + attention improved
> the *distribution* of actions but didn't shift the *mode* to offensive.

## Key Findings

1. **Kill density reversal:** G4 (0.00361) breaks the G1→G2→G3 downward trend, a 29% recovery from G3 (0.00279). The first upward movement in the ablation.
2. **Preservation reward is non-zero:** Mean 0.68 per episode — confirms team reward mechanism is active and flowing between agents.
3. **Episode length stabilized:** 187 steps (vs G3's 179) — slight survival improvement, not dramatic.
4. **Return recovery:** Mean +0.53 (vs G3's 0.15, G2's 0.30) — the best return since G1, driven by both kills and team reward.
5. **Deterministic policy still passive:** The policy mode hasn't shifted. The reward landscape still makes passivity the safest argmax action. The team reward and attention improved exploration quality but not the greedy policy.
6. **Knight role undeveloped:** Knights turn in place 96% of the time. The "Guardian" hypothesis (knights intercepting for archers) didn't emerge at this training duration/architecture.
7. **Archer role partially developed:** Archers shoot 8.6% stochastically (up from 0% in G3), showing the team reward incentivizes risk-taking when sampled, even if not as the argmax action.
