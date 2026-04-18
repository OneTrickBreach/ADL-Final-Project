# 15-Minute Presentation Guide
## "Structured Teamwork in Multi-Agent PPO: A 7-Game Study of Coordination Primitives"

> **Format:** 15 min talk + 5 min Q&A
> **Audience:** Deep-learning / MARL-aware
> **Tone:** Positive-result story; honest about the G5 knight-passivity limitation.

---

## Title slide (0:00–0:30)

**Title:** "From Unrestricted Chaos to Structured Teamwork"
**Subtitle:** How much cooperative performance comes from *structure* vs. *reward shaping*?
**Visual:** `results/v3/demo_sidebyside_v3.mp4` thumbnail (G0 heuristic left, G5 learned right).

> *"We ran a 7-game controlled experiment on PettingZoo KAZ. Every game uses the
> same 30-second episode, same spawn rate, and (critically) the same zombie seeds
> at evaluation time. We ablate four coordination primitives and ask which one
> gives you the most lift."*

---

## Slide 1 — The question (0:30–1:30)

**What changes if you make cooperation *structural* instead of *reward-engineered*?**

- V1/V2 of this project tried to coax cooperation from reward shaping (team blend,
  death penalty, fog + GRU). We hit a pathological regime called *deterministic
  passivity*.
- V3 changes the question: instead of punishing bad behaviour, **build coordination
  into the environment** and measure what the policy can pick up.
- Four primitives, one at a time: ammo discipline → role assignment → target
  locks → pragmatic override.

> *"One-sentence takeaway: the single biggest win is roles; adding more rules on
> top actually hurts."*

---

## Slide 2 — Environment & setup (1:30–2:30)

- KAZ, PettingZoo Butterfly. 2 archers + 2 knights vs. zombies spawning top-of-screen.
- Episode = **450 steps @ 15 FPS = 30 s**.
- Observation per agent = `(27, 5)` entity vectors + **5 V3 extras**
  (role, in-end-zone, ammo %, stamina %, has-lock) → flat 140-dim.
- Score = **kills − failures** (failure = zombie crosses the bottom).
- Evaluation seeds `42, 43, …, 51` used identically across every game so score
  differences isolate the primitive, not the scenario.

**Visual:** KAZ screenshot with an overlay line at y = 0.80·H marking the end zone.

---

## Slide 3 — The 7-game ladder (2:30–4:00)

| Game | Learner? | Primitive added                                          |
|------|----------|----------------------------------------------------------|
| G0   | heuristic | baseline (no ammo, no stamina)                           |
| G1a  | heuristic | global ammo pool + knight stamina + immobile archers     |
| G1b  | heuristic | individual ammo pools + knight stamina + immobile archers|
| G2   | MAPPO (attention) | learned teamwork under the winning ammo mode     |
| G3   | MAPPO + GRU       | +roles (1 forward / 1 end-zone patrol knight)    |
| G4   | MAPPO + GRU       | +target locks (knights lock within radius; archer skips) |
| G5   | MAPPO + GRU       | +pragmatic override (archer overrides lock if zombie will cross first) |

**Visual:** 7 icons along the bottom, each lighting up the new rule.

> *"Games G0–G1b are scripted; they give us floor and ceiling. Games G2–G5 are
> transfer-trained, each from the previous checkpoint."*

---

## Slide 4 — Architecture (4:00–5:00)

```
obs(27×5) ─► EntityAttentionEncoder ─► (optional GRUCell) ─► MLP ─► π / V
                    +
            obs_extras(5) ─► Linear ─► add to pooled vector
```

- Shared-parameter MAPPO across all 4 agents (role is an input feature, not a
  separate network).
- G2: attention only (432 k params). G3/G4/G5: attention + GRU (827 k params).
- Transfer-learning: G3 gets 22 / 26 tensors from G2; G4 and G5 each transfer
  32 / 32 from their predecessor.

**Visual:** block diagram of the network.

---

## Slide 5 — Headline numbers (5:00–7:00)

**Visual:** `results/v3/score_evolution.png` — 7 bars, G3 tallest of the learned set.

| Game | Stochastic score | Deterministic score |
|------|------------------|---------------------|
| G0   | 10.10 ± 4.11     | —                   |
| G1a  | 3.70 ± 1.19      | —                   |
| G1b  | 3.70 ± 1.19      | —                   |
| **G2** | 2.40 ± 0.66    | 0.80 ± 0.75         |
| **G3** | **4.70 ± 1.49** | 2.40 ± 1.28       |
| **G4** | 3.90 ± 0.70    | **2.70 ± 1.27**     |
| **G5** | 3.50 ± 1.28    | 2.60 ± 0.92         |

**Three observations:**
1. **G0 is the real ceiling** — but the constraints below G0 are realistic
   (ammo/stamina, bounded action budgets).
2. **G3 is peak learned performance**: roles alone lift score by **+96 %** over G2.
3. **Deterministic scores improve G2 → G4** (0.80 → 2.70) — structure produces
   committed argmax behaviour, reversing V1/V2's deterministic passivity.

---

## Slide 6 — Why roles help so much (7:00–8:30)

**Visual:** `results/v3/metric_breakdown.png` (4-panel: ammo %, stamina %, kills A/K, attacks A/K).

- G2 archer kills 2.3, knight kills 0.1. The knight attacks 56× but has no
  spatial anchor — it runs after whatever zombie is closest and misses.
- G3 archer kills 4.4, knight kills 0.3. Attack budget drops to 28.9 per knight
  (vs. 56) because the patrol knight is **guarding a line**, not chasing.
- Role features + GRU memory let the policy condition "what am I here to do?"
  on its slot identity.

> *"Roles convert 'chase the nearest zombie' into 'hold the line / chase the rest'.
> That's where the coordination gain comes from."*

---

## Slide 7 — When more rules hurt (8:30–10:00)

**Panel:** knight-attacks column of `metric_breakdown.png` drops to **0–1 per episode**
in G4/G5.

- Lock radius = 15 % of screen width ≈ 192 px. The forward knight in G4 can only
  attack a zombie inside that circle, and the patrol knight must also stay inside
  the end-zone band.
- Result: G4/G5 knights become passive spectators; archers do all the kills.
- This is a **failure-mode finding**, not a negative result: it tells us exactly
  what to relax in a next iteration (wider lock radius, lower stamina cost, or a
  learned role-bonus instead of hard-coded).

> *"We'd rather surface this than hide it. It's the cleanest demonstration that
> locks — a coordination primitive we expected to help — actually gate learning."*

---

## Slide 8 — Why we pivoted from V2 (10:00–10:45)

**Honest slide.**

- V1/V2 narrative = "deterministic passivity from penalty stacking; GRU recovers it."
- That story needed 5 games just to set up the failure mode, and the GRU fix was
  worth only a 25 % kill-density gain.
- V3 re-frames the experiment around **coordination primitives**. Same environment,
  but now every game adds positive structure. You get a cleaner progression and
  an actionable finding (roles > locks).

---

## Slide 9 — Saliency / attention (10:45–11:45)

**Visual:** `results/v3/saliency_v3.png` — attention weights for G3 (left) vs G5 (right).

- G3 attention concentrates on a handful of entity slots (the nearest zombies and
  the other agent).
- G5 attention is more diffuse — consistent with the pragmatic override: the
  archer has to keep track of knight locks as well as nearby zombies.

---

## Slide 10 — Reproducibility (11:45–12:30)

```bash
bash scripts/mega_train_v3.sh                     # 3.5 h on an RTX 5070 Ti
for g in g0 g1a g1b; do ./.venv/bin/python src/evaluate_v3.py --game $g --episodes 10; done
for g in g2 g3 g4 g5; do
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3/$g/final.pt --episodes 10
  ./.venv/bin/python src/evaluate_v3.py --game $g --checkpoint models/v3/$g/final.pt --episodes 10 --deterministic
done
./.venv/bin/python src/phase_artifacts_v3.py
```

- Every evaluation seed is `42 + episode_index`, fixed across games.
- CUDA / MPS / CPU auto-detected; `rules.md` enforces `./.venv/bin/python`.

---

## Slide 11 — Limitations & future work (12:30–13:30)

- **Knight passivity under tight locks.** Widen the radius (say 25 %) or split
  the policy per role (~4× params).
- **Zero failures in every game.** At the default spawn rate the environment is
  too easy — the failure penalty never fires. Next study: sweep spawn_rate and
  report the pressure level at which each game fails.
- **Single ammo-mode tie-break.** G1a and G1b tied exactly (3.70 ± 1.19); the
  orchestration script deterministically picks "global". Re-running with
  different seeds might swap the winner.

---

## Slide 12 — Take-aways (13:30–14:30)

1. Structure beats reward engineering. **Roles alone give +96 % score** over a
   constrained baseline.
2. More constraints are not free: locks produced knight passivity. The
   ablation design lets us attribute that loss to a specific primitive.
3. The evaluation protocol (identical seeds across games) makes these claims
   falsifiable; the repo ships with everything needed to reproduce.

---

## Slide 13 — Demo & Q&A (14:30–15:00)

Play `results/v3/demo_sidebyside_v3.mp4` (G0 unrestricted chaos vs. G5 learned
discipline) while taking questions.

**Likely questions & pocket answers:**

- *"Why is G0 best-in-class?"* — it has no ammo/stamina cap; it's the ceiling,
  not a fair competitor. G3 is the best *constrained* policy.
- *"Why not just train G0?"* — the pseudoplan's scientific question is about
  coordination under structure, not about raw performance.
- *"Did you try a larger lock radius?"* — no; that's exactly the follow-up flagged
  on Slide 11.
- *"Why transfer-learning?"* — G3–G5 share architecture; initialising from the
  previous game's policy dramatically speeds up adaptation (and is fair because
  the obs_dim is identical across all V3 games).
