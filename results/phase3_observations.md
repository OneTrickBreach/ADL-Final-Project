# Phase 3 — Behavioral Observations (Game 4: 60/40 Comrade Healthcare + Attention)

## Summary

Game 4 introduced two simultaneous changes: **team reward** (60/40 self/team split) and **entity self-attention** encoder. The result is a **partial recovery** — stochastic performance improved significantly over G3, but the deterministic policy remains passive.

## What Improved

### Kill Rate Recovery
- Raw kill density: 0.00279 (G3) → 0.00361 (G4) = **+29%**
- This is the first upward movement in the ablation after two consecutive drops
- Mean raw kills per episode: 0.50 (G3) → 0.68 (G4)

### Return Recovery
- Mean return: 0.15 (G3) → 0.53 (G4) = **+253%**
- This includes both kill rewards and preservation rewards
- The preservation component alone contributes +0.68 per episode (40% × team kills)

### Archer Offensive Actions
- Stochastic shoot rate: 0.0% (G3) → 8.6% (G4)
- The team reward provides enough positive signal for archers to risk shooting when sampled
- Archer action distribution is now diverse (62% move, 24% turn, 9% shoot, 5% noop)

## What Didn't Improve

### Deterministic Policy Still Passive
- Deterministic mean return: 0.00 ± 0.00 (same as G2 and G3)
- Deterministic archer shoot %: 0.0% (unchanged from G3)
- The policy's **mode** (argmax) is still a passive action in all states
- Kills come from stochastic sampling, not intentional offensive strategy

### Knight Role Not Developed
- Knights turn in place ~96% of the time (both stochastic and deterministic)
- Knight attack rate: 0.0% in both modes
- The "Guardian" hypothesis (knights moving to protect archers) did not emerge
- Possible reasons: knight movement cost (0.006 after weighting) accumulates faster than the sparse team reward signal (0.133 per rare kill event)

### Episode Length Barely Changed
- 179 steps (G3) → 187 steps (G4) = only +4.5%
- Team coordination didn't meaningfully extend survival
- Agents still die relatively quickly despite team reward

## Analysis: Why Partial Recovery?

### The Stochastic-Deterministic Gap
The gap between stochastic (+0.53) and deterministic (0.00) performance reveals the core issue: **the policy learned that offensive actions have positive expected value under the new reward, but the argmax action is still passive.**

This happens because:
1. The kill probability per shoot attempt is low (many misses)
2. The penalty for missing/dry-fire (−0.3 after weighting) is immediate and certain
3. The kill reward (+0.6 for self) is high but rare
4. The mode of the action distribution is the action with highest average reward across all states — passivity is "safest on average"

### Attention Impact
The attention mechanism enables entity-level reasoning, but with only 500K training steps, the attention patterns may not have fully converged to meaningful entity relationships. The entropy dropped to 0.78 (from max 1.79), suggesting the policy is converging but still has exploration capacity.

### Team Reward Impact
Team reward clearly works — preservation reward is +0.68 per episode, meaning agents receive cooperative bonus. But the bonus is sparse (only when a teammate kills) and doesn't directly reward the knight's protective movement.

## Implications for Phase 4 (Game 5)

The G4 results suggest G5 faces challenges:
1. **The deterministic policy problem persists** — G5's GRU memory won't fix this if the reward landscape still favors passivity
2. **Fog will make kills even rarer** — reduced visibility means fewer shots and lower team reward
3. **Longer training may help** — entropy at 0.78 suggests the policy hasn't fully converged; 750K-1M steps for G5 might allow the attention + GRU to develop more offensive patterns
4. **The hero narrative depends on G5 breaking through** — if the deterministic policy remains at zero, the ablation story becomes "constraints dominate regardless of architecture," which is still a valid but less dramatic finding

## The Honest Ablation Narrative

```
G1: MLP → Greedy Soldier (kills freely, high density)
G2: MLP + Ammo → Risk Avoider (stops shooting to avoid penalty)
G3: MLP + Stamina → Fully Passive (stops all action)
G4: Attention + Team → Recovering Cooperator (kills recover stochastically, team reward flows,
                       but greedy policy still passive)
G5: Attention + GRU + Fog → ??? (can memory and attention overcome fog + all penalties?)
```
