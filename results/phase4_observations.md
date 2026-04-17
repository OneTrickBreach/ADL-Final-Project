# Phase 4 Observations — Game 5 (Tactical Uncertainty) — V2: Death Penalty

> Checkpoint: `models/game5/final.pt`  
> Trained: 2,007,048 steps / 2,764 episodes (V2: 2M steps, death penalty 2.0)  
> Arch: `attention_gru` (825,481 params)  
> Transfer: 26/26 param tensors from G4 `attention` checkpoint  

---

## 1. Training Dynamics

### Entropy Trajectory

| Update | Step | Entropy | Note |
|--------|------|---------|------|
| 5 | 40K | 1.55 | Warm start from G4 (not full reset) |
| 30 | 243K | 0.89 | First positive aggr (+0.0022) |
| 85 | 688K | 0.68 | Stable regime |
| 130 | 1049K | 0.55 | Policy tightening |
| 145 | 1168K | 0.61 | Peak performance: R=+0.0034 |
| 185 | 1491K | 0.79 | Final: entropy slightly recovered |

Entropy settled in the **0.55–0.79** range throughout training — healthy exploration maintained.  
The minimum (0.55) corresponds to peak aggression (+0.0030). No entropy collapse.

### Reward Signal Breakdown

**Phase 1 (Updates 1–24): Warm-up from G4**
- `aggr` near zero or slightly negative (inherited G4 cautious policy)
- `pres` consistently positive (~+0.002–0.004)

**Phase 2 (Updates 25–50): GRU kicks in**
- `aggr` goes positive for the first time at update 25 (+0.0002)
- Update 30: aggr=+0.0022 — aggressive behavior emerging despite fog
- Evidence that GRU temporal memory is tracking entity positions through noise

**Phase 3 (Updates 50–185): Stable "Fire Discipline"**
- Both `aggr` and `pres` consistently positive
- Policy oscillates but never returns to G4-style passivity
- Peak: update 145 → R=+0.0034, aggr=+0.0030, pres=+0.0040

---

## 2. Key Finding: Death Penalty + GRU Synergy

The central failure of G3 and G4 was **deterministic passivity** — the policy mode (argmax action) defaulted to noop or movement, never attacking.

**V2 experiment: does death penalty (−2.0) alone fix this?**

| Game | V2 Stoch Raw Kills | V2 Det Raw Kills | Death Penalty Fixed? |
|------|-------------------|-----------------|---------|
| G3 (MLP) | 0.55/ep | **0.00/ep** | **NO** |
| G4 (Attn) | 0.58/ep | **0.00/ep** | **NO** |
| G5 (Attn+GRU) | 0.68/ep | **0.33/ep** | **PARTIALLY** |

**Answer: No for MLP, Yes for GRU.** Death penalty alone does NOT fix passivity in MLP models. But combined with GRU, it produces the strongest constrained-game performance:
- V2 G5 raw kill density: **0.00373** (+25% vs V1's 0.00299)
- V2 G5 is now the most aggressive constrained game

| Eval Mode | V2 Mean Return | Raw Kill Density | Preservation |
|-----------|----------------|-----------------|---------------|
| G5 Stochastic | −2.07 ± 0.38 | 0.00373 | +0.675 |
| G5 Deterministic | −2.06 ± 0.24 | 0.00182 | +0.325 |

### Why GRU + death penalty works:

1. **Temporal integration under fog:** Individual observations are noisy (σ=0.3), but the GRU accumulates evidence over timesteps. The death penalty provides an *additional gradient signal* that the GRU can act on: "dying is costly, so the temporal evidence about zombie positions should trigger earlier commitment to attack."

2. **MLP capacity bottleneck:** MLP agents receive the same death penalty signal but lack the representational capacity to convert "don't die" into "fight back." They can only learn simple mappings (observation → action) without temporal context.

3. **Warm transfer from G4:** The attention encoder and policy heads started from G4 weights, providing a stable feature space. The GRU learned to augment, not replace, the attention-derived representation.

---

## 3. Behavioral Analysis

**V2 evaluation patterns** (10-episode stochastic eval):

- Mean raw kills: 0.675/ep, mean return: −2.07 ± 0.38
- All 4 agents die in every episode (mean_death_penalty = −2.00), costing −8.0 total per episode
- Despite universal death, G5 is the only constrained game with non-trivial deterministic kills (0.33/ep)

**Failure mode:** All episodes end with team wipeout (ep_len ~181 avg, well below max_cycles=900). Fog-induced misjudged positioning leads to early deaths. The death penalty doesn't prevent dying — it incentivizes fighting before dying.

**Pattern:** Fog creates a bimodal outcome distribution — either agents integrate temporal memory and accumulate kills before death, or early fog-induced errors snowball to quick wipeout. The GRU reduces (but doesn't eliminate) these catastrophic failures. The death penalty amplifies the reward difference between "die having fought" vs "die passively."

---

## 4. Comparison to G4 (V2)

| Metric | G4 V2 (Attention) | G5 V2 (Attn+GRU) | Δ |
|--------|-------------------|-------------------|---|
| Mean return (stoch) | −1.62 | −2.07 | Fog + more deaths |
| Raw kill density | 0.00298 | **0.00373** | **+25%** ↑ |
| Preservation (stoch) | +0.575 | **+0.675** | +17% ↑ |
| Det. raw kills | 0.00/ep | **0.33/ep** | GRU enables action |

> V2 G5 now has HIGHER raw kill density than G4 — the death penalty + GRU combination produces more aggressive behavior than attention alone, even under fog.

---

## 5. Architecture Impact

**Transfer loading (26/26 tensors from G4):**
- Entity attention encoder (8 tensors) — warm start: agents already understand entity structure
- Backbone linear (2 tensors) — adapted during training to GRU output distribution
- Policy/value/decomp heads (16 tensors) — adapted quickly from G4 behavior prior

**GRU parameters (4 tensors, randomly initialized):**
- `weight_ih` (768×256): input-to-hidden projection for 3 gates
- `weight_hh` (768×256): hidden-to-hidden recurrent weights
- `bias_ih`, `bias_hh`: gate biases
- Learning these from scratch took ~200K steps (until aggr went positive at update 25)

---

## 6. Ablation Narrative — V2 Complete

| Game | Behavior Label | Key Mechanism | V1 Kill Density | V2 Kill Density | Δ |
|------|---------------|---------------|----------------|----------------|---|
| G1 | **Greedy Soldier** | Unconstrained kill reward | 0.01089 | 0.01051 | −3.5% |
| G2 | **Risk Avoider** | Ammo penalty suppresses archer | 0.00431 | 0.00308 | −29% |
| G3 | **Fully Passive** | Stamina cost dominates policy mode | 0.00279 | 0.00297 | +6.5% |
| G4 | **Recovering Cooperator** | Team reward + attention | 0.00361 | 0.00298 | −17% |
| G5 | **Fire Discipline** | GRU + death penalty synergy | 0.00299 | **0.00373** | **+25%** |

The full V2 ablation confirms: **death penalty is complementary to architecture, not a substitute.** The only game where it produces a measurable positive effect is G5, where GRU temporal memory provides the capacity to use the survival signal.

---

## 7. Artifacts (V2)

| Artifact | Path |
|----------|------|
| Final checkpoint | `models/game5/final.pt` |
| Intermediate checkpoints | `models/game5/checkpoint_*.pt` |
| Stochastic eval (10 ep) | `results/game5_eval_results.json` |
| Deterministic eval (10 ep) | `results/game5_eval_results_det.json` |
| Demo videos | `results/game5_demo/episode_*.mp4` |
| TensorBoard logs | `results/tensorboard/game5/` |
| Baseline metrics | `results/game5_baseline_metrics.json` |
| Ablation table (V2) | `results/final_ablation_table.md` |
| V1 backup | `models_v1_backup/` |
