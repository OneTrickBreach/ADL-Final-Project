# Phase 2 Behavioral Observations

## Game 2 — Ammo Restriction (Trigger Discipline)

### Training Signal
- Archer starts with 15 arrows per episode. Once depleted, every attempted shot incurs a −0.5 dry-fire penalty.
- During training, early episodes show negative aggression spikes as archers exhaust ammo and keep firing blanks.
- By ~200k steps, the aggression reward stabilizes positive, suggesting archers learn to conserve ammo.

### Observed Behavior
- **Trigger discipline partially emerged:** Archers no longer spam action 4 (shoot) continuously. Return variance is low (±0.28), indicating consistent but cautious play.
- **Kill density dropped 3.7×** from G1 baseline (0.01039 → 0.00279). Fewer shots means fewer kills, but each shot is more deliberate.
- **Episode length halved** (534 → 197). With limited offense, zombies accumulate and overwhelm the team faster.
- **Knights unaffected:** No stamina cost yet, so knights behave similarly to G1 — charging and slashing freely.

### Emergent Pattern
> The archer transitions from a "spray and pray" shooter to a more **selective marksman**. However, 15 arrows may still be generous enough that the constraint is soft. Consider reducing to 8–10 in future tuning.

---

## Game 3 — Ammo + Stamina Decay (Economic Positioning)

### Training Signal
- Knights now pay 0.01 per movement and 0.05 per sword swing, deducted directly from aggression reward.
- Combined with the archer ammo limit from G2, both agent types face resource pressure.
- Training entropy collapsed from 1.79 → 0.83, indicating a highly deterministic policy.

### Observed Behavior
- **Economic positioning partially emerged:** The knight's policy converged sharply — entropy nearly halved compared to G2. This suggests the knight learned to minimize unnecessary actions.
- **Kill density dropped another 5.7×** from G2 (0.00279 → 0.00049). The team kills very rarely per timestep.
- **Return near zero** (0.10 ± 0.32) with some episodes going negative. The stamina cost + ammo limit creates a very tight reward budget.
- **Episode length similar to G2** (~201 vs 197), suggesting the bottleneck is team survival against zombies, not stamina exhaustion.

### Emergent Pattern
> The knight shifts from a "chaser" to a more **stationary defender**, reducing movement to conserve stamina. The combined resource pressure forces both agents into a conservative posture. The entropy collapse confirms the agents found a narrow survival strategy rather than exploring diverse tactics.

---

## G2 vs G3 — Impact of Stacked Constraints

| Aspect | Game 2 | Game 3 |
|--------|--------|--------|
| Policy entropy (final) | ~1.50 | ~0.83 |
| Kill density | 0.00279 | 0.00049 |
| Mean return | 0.55 | 0.10 |
| Agent most affected | Archer | Knight (+ Archer) |
| Behavioral shift | Selective shooting | Stationary defense |

The stacking is clearly additive: each constraint layer reduces offensive output and narrows the policy. Game 3's entropy collapse is a potential concern — the policy may be too rigid for Game 4/5 where coordination and adaptation are needed. Increasing `ent_coef` for later games is worth considering.
