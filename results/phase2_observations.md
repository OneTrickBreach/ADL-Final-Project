# Phase 2 — Behavioral Observations

## Game 2: Ammo Restriction (15 arrows/episode, −0.5 dry-fire penalty)

### Expected Behavior: "Trigger Discipline"
The archer should learn to aim carefully and conserve arrows, firing only when a kill is likely.

### Observed Behavior: "Learned Helplessness"
The archer learned that **not shooting is safer than shooting poorly**. Under deterministic evaluation, the shoot action drops to 0.15% of all actions — the policy almost completely suppresses firing. The dominant action becomes forward movement (43.5%) and no-op (33.4%).

Under stochastic evaluation, random sampling occasionally triggers a shot at the right moment, producing 0.85 raw kills per episode. But this is accidental — the learned policy prefers to never shoot.

**Root Cause:** The dry-fire penalty (−0.5) fires for every missed shot attempt when ammo reaches zero. With 15 arrows per episode and sparse kill rewards (+1 per kill, infrequent), the expected penalty from indiscriminate shooting exceeds the expected reward from occasional kills. PPO correctly optimizes for the highest expected return: don't shoot.

**This is not a bug — it's a lesson.** The penalty was too strong relative to the reward signal. A weaker penalty (e.g., −0.1) or ammo refills would push toward trigger discipline instead of trigger avoidance.

### Metric Impact
- Raw kill density: 0.00431 (2.5× drop from G1's 0.01089)
- Episode length: 197 steps (63% shorter than G1's 533)
- Agents die faster because they stop fighting back, not because zombies got harder

---

## Game 3: Stamina Decay (0.01/move, 0.05/attack, stacked on G2 ammo)

### Expected Behavior: "Economic Positioning"
The knight should learn to minimize unnecessary movement and wait at strategic positions.

### Observed Behavior: "Complete Passivity"
Under deterministic eval, the attack action drops to exactly 0%. The dominant actions are turning (69.3%) and forward movement (30.4%). The knight learned that attacking costs stamina (−0.05) and rarely produces kills, so the optimal strategy is to never attack.

Stochastic eval produces 0.50 raw kills per episode, again from random sampling rather than learned behavior.

**Root Cause:** Stacking stamina costs on top of ammo limits creates a compounding penalty environment. The knight's attack (action 4) costs 0.05 per use. With kills being sparse (+1, infrequent), the expected stamina cost over an episode exceeds the expected kill reward. Combined with the archer's shooting suppression from G2, the team has effectively stopped engaging zombies altogether.

### Metric Impact
- Raw kill density: 0.00279 (3.9× drop from G1, 1.5× drop from G2)
- Episode length: 179 steps (further 9% compression from G2)
- Penalty drag: 70% of raw kills are offset by penalty costs in the shaped reward

---

## Cross-Game Analysis

### The Avoidance Cascade
Each constraint layer pushes the policy further from offense:

```
G1: "Kill everything"       → 40% shoot action, 0.01089 kill density
G2: "Don't waste ammo"      → 0.15% shoot action, policy says "don't shoot at all"
G3: "Don't waste stamina"   → 0% attack action, policy says "don't do anything costly"
```

This is a textbook example of **reward hacking through avoidance**: when penalties for failure are stronger than rewards for success, RL agents learn to avoid the penalized action class entirely, rather than learning to perform it well.

### Survival Paradox
Intuitively, conservative agents should survive longer. But episode lengths **decrease** (533→197→179). This happens because:
1. Zombies keep spawning regardless of agent behavior
2. Agents that stop fighting accumulate more zombies on screen
3. More zombies → faster death
4. The agents traded offensive capability for nothing

### What This Means for Phase 3
The 60/40 comrade healthcare reward in Phase 3 won't fix this. Adding team rewards on top of agents that already refuse to act will just add another term to a near-zero reward signal. Before Phase 3, the penalty magnitudes need tuning:
- **Reduce dry-fire penalty:** −0.5 → −0.1 (punish waste without dominating the reward)
- **Add ammo refill mechanics:** Replenish arrows periodically to reward conservation rather than abstinence
- **Scale stamina cost:** Make it proportional to remaining stamina (high stamina = low cost, low stamina = high cost) so early actions are cheap

### Deterministic vs Stochastic Evaluation
The stochastic eval results are superficially encouraging (kills happen, returns are positive). The deterministic eval reveals the truth: the learned policy has zero offensive intent. For future phases, **always check deterministic eval** alongside stochastic to catch policy collapse early.
