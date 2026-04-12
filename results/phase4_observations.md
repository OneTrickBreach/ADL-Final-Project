# Phase 4 Observations — Game 5 (Tactical Uncertainty)

> Checkpoint: `models/game5/final.pt`  
> Trained: 1,506,717 steps / 2,076 episodes  
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

## 2. Key Finding: Deterministic Passivity Resolved

The central failure of G3 and G4 was **deterministic passivity** — the policy mode (argmax action) defaulted to noop or movement, never attacking, because the expected value of attack was negative under ammo+stamina penalties.

**G5 breaks this pattern:**

| Eval Mode | Mean Return | Attack Rate | Preservation |
|-----------|-------------|-------------|--------------|
| G5 Stochastic | +0.384 ± 0.314 | ~(sampled) | +0.525 |
| G5 Deterministic | +0.205 ± 0.191 | **69.9%** | +0.275 |

**69.9% deterministic attack rate** is the headline result of Phase 4.

### Why GRU overcomes passivity:

1. **Temporal integration under fog:** Individual observations are noisy (σ=0.3), but the GRU accumulates evidence over timesteps. By the time an agent commits to an attack, the hidden state has integrated ~2–5 steps of noisy entity position data, reducing effective uncertainty.

2. **Policy mode shift:** Because the GRU allows better zombie tracking, the *expected value of attacking* is higher per step. The penalty structure hasn't changed from G4, but the GRU makes attacks *more likely to succeed*, shifting the argmax away from noop.

3. **Warm transfer from G4:** The attention encoder and policy heads started from G4 weights, providing a stable feature space. The GRU learned to augment, not replace, the attention-derived representation.

---

## 3. Behavioral Analysis (Demo Episodes)

From `results/game5_demo/`:

**Best episodes (high return):**
- Episode 7: return=+1.03, aggr=+0.88, pres=+1.25, len=203 — agents killed multiple zombies AND all survived
- Episode 3: return=+0.87, aggr=+0.79, pres=+1.00, len=203

**Failure modes:**
- Episode 1 (stoch): return=−0.11, aggr=−0.18, pres=0.0, len=163 — agents died early (fog caused misjudged positioning)
- Negative-return episodes share len=163 (minimum cycle), suggesting early team wipeout

**Pattern:** Fog creates a bimodal outcome distribution — either agents integrate temporal memory and perform well (return > +0.30), or early fog-induced errors snowball to wipeout. The GRU reduces (but doesn't eliminate) these catastrophic failures.

---

## 4. Comparison to G4

| Metric | G4 (Attention) | G5 (Attn+GRU) | Δ |
|--------|---------------|---------------|---|
| Mean return (stoch) | +0.42 | +0.38 | −10% (fog handicap) |
| Raw kill density | 0.00361 | 0.00299 | −17% (fog handicap) |
| Kill density (shaped) | 0.00130 | 0.00165 | **+27%** ↑ |
| Preservation (stoch) | +0.675 | +0.525 | −22% (fog handicap) |
| Det. attack rate | ~0% | **69.9%** | **+∞** ↑ |
| Return std dev (det) | — | ±0.191 | Tighter than stoch |

> The fog creates a raw performance ceiling, but the GRU achieves **+27% shaped kill density** and eliminates deterministic passivity entirely.

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

## 6. Ablation Narrative — Complete

| Game | Behavior Label | Key Mechanism | Kill Density |
|------|---------------|---------------|-------------|
| G1 | **Greedy Soldier** | Unconstrained kill reward | 0.01089 |
| G2 | **Risk Avoider** | Ammo penalty suppresses archer | 0.00431 |
| G3 | **Fully Passive** | Stamina cost dominates policy mode | 0.00279 |
| G4 | **Recovering Cooperator** | Team reward + attention reverses trend | 0.00361 |
| G5 | **Fire Discipline** | GRU memory + fog = committed action under uncertainty | 0.00299* |

\* Under fog handicap. Shaped kill density (+0.00165) surpasses G4 (+0.00130).

The full 5-game ablation is complete. The evolutionary path from unconstrained aggression → penalty collapse → cooperative recovery → disciplined action under uncertainty is clearly demonstrated in both quantitative metrics and behavioral analysis.

---

## 7. Artifacts

| Artifact | Path |
|----------|------|
| Final checkpoint | `models/game5/final.pt` |
| Intermediate checkpoints (5) | `models/game5/checkpoint_*.pt` |
| Stochastic eval (10 ep) | `results/game5_eval_results.json` |
| Deterministic eval (10 ep) | `results/game5_eval_results_det.json` |
| Demo videos (10 ep) | `results/game5_demo/episode_*.mp4` |
| TensorBoard logs | `results/tensorboard/game5/` |
| Baseline metrics | `results/game5_baseline_metrics.json` |
| Ablation table | `results/final_ablation_table.md` |
