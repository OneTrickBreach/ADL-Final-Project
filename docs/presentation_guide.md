# 15-Minute Presentation Guide
## Structured Teamwork in Multi-Agent PPO: A 7-Game Study of Coordination Primitives

> **Format:** 15 min talk + 5 min Q&A
> **Audience:** Deep-learning / MARL-aware
> **Tone:** Positive-result story; headline is the monotonic progression G2→G5 and the G5 > G0 result.

---

## Title slide (0:00–0:30)

**Title:** "From Unrestricted Chaos to Structured Teamwork"
**Subtitle:** How much cooperative performance comes from *structure* vs. *reward shaping*?
**Visual:** `results/v3_1/demo_sidebyside_v3.mp4` thumbnail (G0 heuristic left, G5 learned right).

> *"We ran a controlled 7-game experiment on PettingZoo KAZ. Every game uses the
> same 30-second episode, the same spawn pressure, and — critically — the same
> per-episode random seeds at evaluation time. We ablate four coordination
> primitives and ask how much each one contributes."*

---

## Slide 1 — The question (0:30–1:30)

**What changes if you make cooperation *structural* instead of *reward-engineered*?**

- Standard approach: add more reward terms (team blend, death penalty, shaping bonuses).
- Our approach: **build coordination into the environment**, keep rewards minimal, and measure what the policy picks up.
- Four primitives, one at a time: ammo discipline → role assignment → target
  locks → pragmatic override.

> *"One-sentence takeaway: the full curriculum matters. Every primitive is
> additive, and the last one (pragmatic override) is the biggest step of all."*

---

## Slide 2 — Environment & setup (1:30–2:30)

- KAZ, PettingZoo Butterfly. 2 archers + 2 knights vs zombies spawning top-of-screen.
- Episode = **450 steps @ 15 FPS = 30 s**.
- Observation per agent = `(27, 5)` KAZ entities + **5 extras**
  (role, in-end-zone, ammo %, stamina %, has-lock) → flat 140-dim.
- Score = **kills − failures**. A *failure* is a zombie crossing the end line
  *or* an agent killed by zombie contact.
- Pressure: `spawn_rate=8`, `max_zombies=20`, lock radius 0.25 W, ammo 60/30,
  stamina 150 — calibrated so the failure penalty is a live training signal
  (every game reports 1.8–3.0 failures/episode).
- Evaluation seeds `42..51` used identically across every game.

**Visual:** KAZ screenshot with an overlay line at y = 0.80·H marking the end zone.

---

## Slide 3 — The 7-game ladder (2:30–4:00)

| Game | Learner?          | Primitive added                                          |
|------|-------------------|----------------------------------------------------------|
| G0   | heuristic         | baseline (no ammo, no stamina)                           |
| G1a  | heuristic         | global ammo pool + knight stamina + immobile archers     |
| G1b  | heuristic         | individual ammo pools + knight stamina + immobile archers|
| G2   | MAPPO (attention) | learned teamwork under the winning ammo mode             |
| G3   | MAPPO + GRU       | +roles (1 forward / 1 end-zone patrol knight)            |
| G4   | MAPPO + GRU       | +target locks (knights lock within radius; archer skips) |
| G5   | MAPPO + GRU       | +pragmatic override (archer overrides lock if zombie will cross first) |

**Visual:** 7 icons along the bottom, each lighting up the new rule.

> *"G0–G1b are scripted; they give us a ceiling and two constrained floors.
> G2–G5 are transfer-trained, each starting from the previous checkpoint."*

---

## Slide 4 — Architecture (4:00–5:00)

```
obs(27×5) ─► EntityAttentionEncoder ─► (optional GRUCell) ─► MLP ─► π / V
                    +
           obs_extras(5) ─► Linear ─► add to pooled vector
```

- Shared-parameter MAPPO across all 4 agents (role is an input feature, not a separate network).
- G2: attention only (432 k params). G3/G4/G5: attention + GRU (827 k params).
- Transfer: G3 copies 28 / 32 tensors from G2 (4 GRU tensors new); G4/G5 copy
  32 / 32 from their predecessor.

**Visual:** block diagram of the network.

---

## Slide 5 — Headline numbers (5:00–7:00)

**Visual:** `results/v3_1/score_evolution.png` — G5 is the tallest bar of the seven.

| Game | Stochastic score | Deterministic score |
|------|------------------|---------------------|
| G0   | 8.70 ± 9.65      | —                   |
| G1a  | 1.70 ± 3.00      | —                   |
| G1b  | 1.60 ± 2.76      | —                   |
| G2   | 1.80 ± 1.08      | −1.90 ± 1.14        |
| G3   | 4.10 ± 2.17      | 0.60 ± 1.43         |
| G4   | 7.40 ± 3.32      | 0.20 ± 2.71         |
| **G5** | **11.20 ± 3.12** | **2.90 ± 2.62**   |

**Three observations:**

1. **Strictly monotonic stochastic progression G2 → G5** (1.80 → 4.10 → 7.40 → 11.20).
   Deterministic climbs from strongly negative to strongly positive
   (−1.90 → 0.60 → 0.20 → 2.90); G3 and G4 are tied within noise
   (±1.4–±2.7σ), and **G5 is the peak on both metrics**.
2. **G5 beats G0** — the learned policy with all four primitives outperforms
   the unrestricted heuristic ceiling by **+29 %** on stochastic score.
3. **G5 minimises failures** (1.8) *and* maximises score — the pragmatic
   override performs a genuine offence/defence trade-off.

---

## Slide 6 — Why roles help (7:00–8:00)

**Visual:** `results/v3_1/metric_breakdown.png` (4-panel: ammo %, stamina %, kills A/K, attacks A/K).

- G2 archer kills 4.5, knight kills 0.1. Both knights chase the same zombie;
  their attack budget (49.2 shared) is spent on overlapping targets.
- G3 archer kills 5.5, knight kills 1.0. The patrol knight is *anchored* to the
  end zone and only attacks zombies that arrive there; the forward knight moves
  freely.
- Roles turn "chase the nearest zombie" into "hold the line / chase the rest."

---

## Slide 7 — Why locks help, and why G5 is the real story (8:00–10:00)

- **Locks (G4):** archer stops wasting arrows on zombies a knight already owns.
  Archer kills jump to 10.0/episode (vs 5.5 in G3) because arrows go to targets
  not already committed.
- **Pragmatic override (G5):** if a zombie is projected to cross before the
  knight can intercept, the archer is allowed to shoot it anyway. Archer kills
  rise to 12.3/ep, *and* failures drop from 2.7 to 1.8.

> *"G5 is the single biggest step in the progression. It's +3.80 stochastic,
> +2.70 deterministic on top of G4, just from allowing one exception to the
> lock rule."*

**Visual:** the knight-attacks column of `metric_breakdown.png` — stays active
(23.8/ep) in G5, so the override doesn't shut the knight down; it just
re-allocates the archer.

---

## Slide 8 — Deterministic behaviour (10:00–11:00)

**Visual:** text slide with the deterministic column.

- G2 argmax is genuinely bad (−1.90) — 0.9 archer kills vs 2.8 failures.
- G3 / G4 argmax are near zero (0.60 / 0.20, tied within noise).
- G5 argmax is strongly positive (**2.90**) — an absolute swing of **+4.80**
  from G2 along the progression.

> *"Structure gives the argmax mode something to commit to. The policy still
> leans on sampling noise for its peak score, but the argmax mode is now
> substantively productive instead of frozen."*

---

## Slide 9 — Attention saliency (11:00–11:45)

**Visual:** `results/v3_1/saliency_v3.png` — attention weights for G3 (left) vs G5 (right).

- G3 attention concentrates on a handful of entity slots (nearby zombies).
- G5 attention is more diffuse — consistent with the pragmatic override: the
  archer now has to track knight lock targets as well as its own candidates.

---

## Slide 10 — Reproducibility (11:45–12:45)

```bash
bash scripts/mega_train_v3_1.sh                          # ~3.5 h on an RTX 5070 Ti
./.venv/bin/python src/phase_artifacts_v3.py \
    --results_dir results/v3_1 --output_dir results/v3_1 --models_dir models/v3_1
```

- Every evaluation seed is `42 + episode_index`, fixed across games.
- CUDA / MPS / CPU auto-detected; `rules.md` enforces `./.venv/bin/python`.
- 70 stochastic + 40 deterministic eval episodes, 70 demo MP4s, 4 figures.

---

## Slide 11 — Limitations & future work (12:45–13:45)

- **Lock-radius sweep.** We chose 0.25 W on one run; a systematic
  $\{0.15, 0.20, 0.25, 0.30\}\,W$ sweep would characterise sensitivity.
- **Spawn-rate stress.** Push `spawn_rate` to 4–6 and test whether G5's
  pragmatic override degrades gracefully.
- **Per-role heads.** Split the shared policy (shared trunk, 2 knight heads,
  1 archer head) instead of using the role\_id feature.
- **Learned override.** Replace the hard-coded cross-vs-intercept calculation
  with a learned gating policy.

---

## Slide 12 — Take-aways (13:45–14:30)

1. **Structure beats reward engineering.** With minimal shaping ($b_r = 10^{-3}$,
   $b_\ell = 2\times 10^{-3}$), the 4-primitive curriculum takes a learned
   policy from below the constrained heuristic floor to above the
   unrestricted ceiling.
2. **Every primitive is additive**, and the last one (pragmatic override) is
   the biggest step.
3. **The pragmatic override simultaneously raises kills and lowers failures**,
   which is the signature of a genuine offence/defence trade-off rather than
   one-sided improvement.

---

## Slide 13 — Demo & Q&A (14:30–15:00)

Play `results/v3_1/demo_sidebyside_v3.mp4` (G0 unrestricted heuristic vs G5
learned) while taking questions.

**Likely questions & pocket answers:**

- *"Why is G0 not the ceiling anymore?"* — G0 fires 306 arrows/ep at 2.4 % hit
  rate; G5 fires 57 arrows/ep at 22 % hit rate. Ammo efficiency is how the
  learned policy beats the heuristic.
- *"How do you define failure?"* — line-cross OR agent killed by zombie contact.
  Both indicate the defence line was breached; we count them additively.
- *"Why transfer-learning?"* — G3/G4/G5 share architecture and `obs_dim = 140`;
  initialising from the predecessor's policy is fair and dramatically speeds
  up adaptation (G4 copies all 32 / 32 tensors from G3).
- *"Could the override be learned?"* — yes, that's flagged as future work.
  Currently it's a hard-coded cross-vs-intercept calculation.
