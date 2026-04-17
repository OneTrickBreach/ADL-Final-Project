# 15-Minute Presentation Guide
## "Shaping Emergent Behavior in Multi-Agent PPO: A Five-Stage Ablation on KAZ"

> **Format:** 15 min talk + 5 min Q&A  
> **Audience:** Deep learning / RL audience with basic MARL familiarity  
> **Tone:** Results-first, narrative-driven, avoid implementation minutiae  

---

## Slide-by-Slide Breakdown

### Slide 1 — Title (0:00–0:30)
**Title:** "From Greedy Soldier to Fire Discipline"  
**Subtitle:** A 5-stage ablation of reward shaping + architecture in cooperative MARL  
**Visual:** Side-by-side thumbnail of `demo_sidebyside.mp4` frame (G1 chaotic vs G5 disciplined)  

> **Speaking notes:** "This project asks one question: when you progressively
> add real-world constraints to a MARL environment, what happens to agent behavior?
> The answer is a journey from a mindlessly aggressive soldier to a disciplined
> fire team — and understanding what triggers each transformation."

---

### Slide 2 — The Problem (0:30–1:30)
**Title:** Why does reward shaping fail?  
**Content:**
- MARL agents optimise reward, not intent
- Stacking constraints (ammo, stamina, fog) → unintended passivity
- Classic failure: agents discover "doing nothing" minimises penalties
- **Key tension:** Safety constraints vs. necessary aggression  

**Visual:** Simple diagram — reward signal components stacking up, agent behavior arrow pointing down

> **Speaking notes:** "Every safety constraint you add is a potential trap. The agent
> doesn't know *why* it's getting penalised — it just knows that attacking sometimes
> costs more than it returns. So it learns not to attack."

---

### Slide 3 — Environment & Setup (1:30–2:30)
**Title:** Knights-Archers-Zombies (PettingZoo)  
**Content:**
- 4 cooperative agents: 2 knights (melee) + 2 archers (ranged)
- Shared-parameter MAPPO — all agents, one network
- Observation: 27 entity slots × 5 features = 135-dim vector
- Action space: 6 discrete actions (move, rotate, attack, no-op)

**Visual:** Screenshot or diagram of KAZ environment + observation structure table

> **Speaking notes:** "Our environment is KAZ — it's clean, fast, and has a natural
> aggression-preservation tension built in. One network shared by all four agents.
> Each agent sees the 27 nearest entities as a structured entity-slot vector."

---

### Slide 4 — The 5-Game Ablation (2:30–4:00)  ⭐ CORE SLIDE
**Title:** Progressive constraint stacking — what changes and why  

| Game | Added Element | Expected Behavior |
|------|--------------|-------------------|
| G1   | Kill reward + death penalty (−2.0) | Aggressive baseline |
| G2   | + Ammo limit (15/ep) | Resource awareness |
| G3   | + Stamina decay | Risk aversion |
| G4   | + 60/40 team reward + Attention | Cooperation |
| G5   | + Gaussian fog + GRU | Fire discipline |

**V2 addition:** Death penalty (−2.0 per death) applied to ALL games. The key experiment: does this alone fix G3 passivity?

**Visual:** Table + constraint stack diagram with color coding matching the ablation chart

> **Speaking notes:** "Each game adds exactly one thing. In V2 we also added a universal
> death penalty — dying costs 2 kills of reward. The question is: does this survival
> incentive alone fix the passivity problem, or do we still need architectural upgrades?"

---

### Slide 5 — The Behavioral Collapse (4:00–5:30)  ⭐ MAIN FINDING SETUP
**Title:** "The Journey: Greedy → Passive → Disciplined"  
**Visual:** `results/kill_density_evolution.png`

**Talking points:**
- G1: Kill density 0.01051 — peak aggression, barely affected by death penalty
- G2→G3: **71–72% drop** in kill density — constraints + death penalty still create passivity
- G4: Flatlines at 0.00298 — attention alone insufficient
- G5: **0.00373** — highest constrained game, +25% vs V1 thanks to GRU + death penalty

> **Speaking notes:** "The collapse is still here — death penalty didn't fix it.
> G2 and G3 agents can't convert 'don't die' into 'fight back' with an MLP.
> But look at G5 — it's now the most aggressive constrained game. The GRU
> gives agents the capacity to *use* the survival signal. That's the key
> V2 finding: reward shaping and architecture are complementary."

**Emphasise:** G5 raw kill density is now HIGHER than G4 — death penalty + GRU is the winning combo.

---

### Slide 6 — The Smoking Gun: Deterministic Passivity (5:30–7:00)  ⭐ KEY FINDING
**Title:** Diagnosing Deterministic Passivity — Death Penalty is NOT Enough  
**Content:**
```
                 V2 Stoch Raw Kills    V2 Det Raw Kills    Death Penalty Effect
G3 (MLP):        0.55/ep               0.00/ep             NO CHANGE
G4 (Attn):       0.58/ep               0.00/ep             NO CHANGE
G5 (Attn+GRU):   0.68/ep               0.33/ep             +25% raw kill density
```

**Explain the mechanism:**
1. Strong penalties shift probability mass *away* from attack
2. Death penalty adds survival incentive but MLP can't convert it to action
3. Argmax policy: always picks the *most probable* action — which is no-op in G2–G4
4. GRU fix: temporal evidence accumulates → policy *mode* shifts to attack AND death penalty amplifies this

**Visual:** V1 vs V2 raw kill density comparison bar chart

> **Speaking notes:** "This is the cleanest ablation in the paper. Same survival
> incentive, same penalty magnitude, applied to all games. MLP agents completely
> ignore it — they lack the representational capacity to convert 'don't die' into
> 'fight back.' But the GRU model *uses* it — V2 G5 has 25% higher raw kill
> density than V1. Architecture is the bottleneck, not reward magnitude."

---

### Slide 7 — Architecture Evolution (7:00–8:30)
**Title:** Why GRU? The architectural choices  
**Visual:** Architecture diagram: MLP → MLP+Attn → Attn+GRU

**Three stages:**
1. **MLP** (G1–G3): Flat observation → 3-layer MLP → actor/critic heads. ~200K params. No temporal memory, no structured entity reasoning.

2. **Entity Self-Attention** (G4): Reshape obs to (27, 5) entity slots → Linear embed → 4-head self-attention with empty-slot masking → mean pool. ~467K params. Permutation-invariant entity reasoning.

3. **Attention + GRU** (G5): Same attention → **GRUCell(256, 256)** → actor/critic. 825K params. Hidden state per agent, reset at episode boundaries. Transfer-loaded 26/26 tensors from G4.

> **Speaking notes:** "Each architectural choice was motivated by a diagnosis.
> The MLP couldn't handle entity structure — enter attention. Attention couldn't
> handle temporal commitment under noise — enter GRU. Transfer learning from G4
> to G5 gave us a warm start and cut early training instability."

---

### Slide 8 — Explainability: Where Do Agents Look? (8:30–10:00)
**Title:** Gradient Saliency (G1) vs. Attention Weights (G5)  
**Visual:** `results/saliency_comparison.png`

**Left panel (G1 saliency):**
- Gradient of chosen-action logit w.r.t. input features
- Concentrated on ~3–4 entity slots (nearest enemies)
- Tunnel vision: ignores teammates entirely

**Centre + Right (G5 attention):**
- Forward hook on `EntityAttentionEncoder.attention`
- Column-wise sum of 27×27 attention matrix over 200 steps
- More distributed across entity slots
- Off-diagonal patterns in heatmap = cross-entity reasoning
- Attention sum = 27.0 (verified: perfect softmax rows)

> **Speaking notes:** "Saliency maps give us a window into *what* the policy
> is paying attention to. G1's MLP has tunnel vision — it fixates on the
> nearest threat. G5's attention is distributed — it's reasoning about
> teammates and threats simultaneously. This is the mechanistic basis for
> cooperative behavior."

---

### Slide 9 — Stress Test: G1 vs G5 Under Pressure (10:00–11:00)
**Title:** Does G5 degrade gracefully?  
**Visual:** `results/collapse_graph.png`

**Key point:** Each model tested on its own training environment at 4 spawn-pressure levels (0.7×, 1×, 1.4×, 2×). Normalised kill density shows G5 degrades less under increased zombie pressure despite operating with 3 additional active constraints (fog, ammo, stamina).

**Honest caveat to mention:** G1 and G5 have different observation dimensions (110 vs 135), preventing a same-environment comparison. This itself reflects how significantly the policies diverged.

> **Speaking notes:** "G5 is playing on hard mode — it has fog, ammo limits, and
> stamina decay, while G1 has none of those. Yet when we normalize to each model's
> own baseline, G5 degrades less under spawn pressure. The GRU's temporal memory
> makes it more robust to sudden increases in threat density."

---

### Slide 10 — Demo (11:00–12:00)
**Title:** Live Demo / Video  
**Visual:** Play `results/demo_sidebyside.mp4`

**What to point out:**
- Left (G1): frantic, disorganized, agents die quickly
- Right (G5): more deliberate movement, coordinated attacks, longer survival
- G5 kill counter climbs despite harder environment

> **Speaking notes:** "This side-by-side captures the behavioral difference.
> G1 attacks everything immediately, wastes resources, gets overrun.
> G5 moves with intent — the GRU hidden state is building a temporal
> model of the threat landscape before committing."

---

### Slide 11 — Conclusion & Takeaways (12:00–13:30)
**Title:** What we learned  

**Four key takeaways:**

1. **Constraint stacking without architecture = passivity trap**  
   Progressive penalties drove a 72% kill-density collapse (G1→G3). Safety constraints need architectural support to avoid defeating the agent's purpose.

2. **Death penalty alone is insufficient**  
   V2's −2.0 death penalty was explicitly designed to make passive dying worse than fighting. MLP agents (G2–G3) completely ignore it. Reward magnitude is not the bottleneck.

3. **Reward + architecture are complementary**  
   Death penalty + GRU produces the strongest constrained-game performance (G5 raw kill density +25% vs V1). Neither works alone; together they unlock active behavior under uncertainty.

4. **Temporal memory resolves the commitment problem**  
   GRU hidden states allow agents to integrate evidence across noisy timesteps and commit to aggressive actions. This was the key architectural ingredient — attention alone was not sufficient.

---

### Slide 12 — Future Work (13:30–14:30)
**Title:** Open questions  

- **Standardise observation dimensions** across all games (enables cross-game transfer evaluation)
- **Communication channels**: does explicit inter-agent messaging replace temporal memory?
- **Centralised training** (MADDPG/QMIX) vs. decentralised MAPPO under same constraint stack
- **Curriculum fog**: gradually increase σ during training rather than full σ=0.3 from step 0
- **Adaptive entropy**: learn entropy coefficient rather than fixed doubling heuristic

---

### Slide 13 — Q&A (14:30–15:00 / overflow into Q&A)
**Pre-empt likely questions:**

**Q: Why not just lower the stamina/ammo penalties in G3?**  
A: We deliberately kept penalties fixed to isolate the architectural contribution. V2's death penalty experiment proves this: even adding a *positive* incentive (survival reward) doesn't fix MLP passivity. The architecture is the bottleneck.

**Q: Why shared parameters? Wouldn't separate networks do better?**  
A: Shared parameters force emergent specialisation — knights and archers must develop different behavior from the same weights, driven only by their different observation contexts. This is more interesting and more parameter-efficient.

**Q: Does the death penalty actually help anything?**  
A: Yes — but only with the right architecture. G5 (GRU) raw kill density improved 25% from V1 to V2. MLP games saw no improvement. This is the key complementarity finding.

**Q: How do you know the GRU is the key ingredient vs. just more parameters?**  
A: Fair challenge. G4 has 467K params, G5 has 825K. A fairer ablation would be a G5 with MLP+Attention+extra capacity but no GRU. This is future work.

---

## Timing Summary

| Section | Slide | Time |
|---------|-------|------|
| Title & hook | 1 | 0:30 |
| Problem setup | 2 | 1:00 |
| Environment | 3 | 1:00 |
| 5-game ablation | 4 | 1:30 |
| Behavioral collapse | 5 | 1:30 |
| Deterministic passivity | 6 | 1:30 |
| Architecture | 7 | 1:30 |
| Explainability | 8 | 1:30 |
| Stress test | 9 | 1:00 |
| Demo | 10 | 1:00 |
| Conclusion | 11 | 1:30 |
| Future work | 12 | 1:00 |
| Q&A buffer | 13 | 0:30 |
| **Total** | | **~15:00** |

---

## Slide Design Tips

- **Font size:** Minimum 24pt for body text, 36pt for titles
- **Dark background** works well (dark navy + white text + the brand colors from the ablation charts)
- **Brand colors:** Use G1–G5 consistent colors throughout:  
  G1=#e74c3c (red), G2=#e67e22 (orange), G3=#95a5a6 (gray), G4=#3498db (blue), G5=#2ecc71 (green)
- **Slides 5, 6, 8** are your three most important — allocate most visual real estate
- **Avoid code on slides** — reference filenames instead (`mappo_net.py`, `EntityAttentionEncoder`)
- **Include slide numbers** for Q&A reference

## Key Numbers to Have Ready

| Fact | V1 | V2 |
|------|-----|-----|
| G1 raw kill density | 0.01089 | 0.01051 |
| G3 raw kill density | 0.00279 | 0.00297 |
| G5 raw kill density | 0.00299 | **0.00373** (+25%) |
| G3 det. raw kills | 0.00/ep | **0.00/ep** (unchanged) |
| G5 det. raw kills | — | **0.33/ep** |
| Death penalty | 0.0 | **2.0** |
| G5 total training steps | 1,506,717 | 2,007,048 |
| G5 parameters | 825,481 | 825,481 |
| Attention sum (validated) | 27.000 | 27.000 |
| Transfer tensors loaded | 26/26 | 26/26 |

## Artifacts for the Presentation

All in `results/`:
- `kill_density_evolution.png` → Slide 5
- `saliency_comparison.png` → Slide 8
- `collapse_graph.png` → Slide 9
- `demo_sidebyside.mp4` → Slide 10
- `final_ablation_table.md` → Reference for Slide 4 table
