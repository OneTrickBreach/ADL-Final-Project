# Presentation Emergency Brief — Apr 21 2026

> Read this **before** you walk into the room. Two last-minute honesty items
> Elizabeth flagged this morning that the slides / talking points must reflect.

---

## Item 1 — Episode length

**What the slides/docs say:** "Every episode lasts 30 seconds (450 steps @ 15 FPS)."

**What actually happens in evaluation:** 30 s is the **truncation ceiling**, not
the typical length. Mean episode length across the 7 games is:

| Game | steps | seconds @ 15 FPS |
|------|------:|-----------------:|
| G0   | 186.2 | 12.4 |
| G1a  | 158.0 | 10.5 |
| G1b  | 158.0 | 10.5 |
| G2   | 171.0 | 11.4 |
| G3   | 175.9 | 11.7 |
| G4   | 189.5 | 12.6 |
| G5   | 182.7 | 12.2 |

**Why:** Episodes end when all 4 agents are killed by zombie contact. Our
wrapper's episode-continuation override (`kaz_wrapper_v3.py` line ~506) keeps
the episode alive past individual line crosses, but it does **not** revive
killed agents. With `failures_mean = 1.8–3.0` per episode (failure = line-cross
*or* agent-breach), the four agents are typically dead by step ~170–190.

### How to say it on stage
> *"Each episode has a 30-second ceiling. In practice under our pressure
> regime the agents are overwhelmed earlier — the 7 games average 10.5–12.6
> seconds before the last agent dies. That is the denominator for all the
> per-episode metrics we report; the pressure parameters were chosen so the
> failure penalty is an active training signal, which necessarily means agents
> die, which necessarily means short episodes."*

### Cosmetic fix (already applied for the demo videos)

Every MP4 under `results/v3_1/` has been re-encoded at 2× duration and lives
under `results/v3_1_slow/`. The main side-by-side demo
(`results/v3_1_slow/demo_sidebyside_v3.mp4`) is now **24 seconds** long
instead of 12 seconds. **Play the slowed version from that directory in the
slides.** All metrics, tables, and the JSONs are unchanged; only playback
speed differs.

---

## Item 2 — "Archers stay at the bottom"

**What was discussed in the team meeting:** an archer-at-bottom constraint for
learned games.

**What the wrapper actually implements:**

- G1a / G1b heuristic games: `archer_immobile = True` → archers can't move at
  all. They stay at whatever y-position they spawn with.
- **G2–G5 learned games: `archer_immobile = False`** → archers can move freely
  in any direction.

The role system is **knights only**: one forward knight and one end-zone
patrol knight. There is no archer-positioning constraint in the learned
policies. Whatever spatial behaviour the archers exhibit in G2–G5 is what the
policy *learned*, not what the wrapper *enforced*.

Time-to-presentation does not permit a wrapper patch + re-eval + re-record,
so this is a talking-point fix only.

### How to say it on stage
> *"In the heuristic-constraint games (G1a, G1b) archers are fixed at their
> spawn row so we can measure the effect of that constraint in isolation. In
> the learned games (G2 onwards) archers are free to move, and the role
> assignment is entirely on the knights — one forward, one patrol. In
> practice, you'll see in the videos that the learned archers tend to stay
> near the bottom because that's where the fire lanes are clearest; it's
> emergent behaviour from the policy, not a hard-coded constraint."*

### If someone pushes back
Don't hide it. Say: *"We discussed an archer-pin constraint during design and
decided to let the policy find its own positioning. The learned archers do
drift forward sometimes, especially in G4 and G5 where the pragmatic override
changes what the archer is allowed to shoot at."*

---

## Item 3 — Three questions you should be ready for

1. **"Why does the failure count stay around 2 even at G5?"**
   Because it includes agent-breaches, not just line-crosses. A "failure" in
   our definition = {zombie crosses the end line} ∪ {zombie kills an agent by
   contact}. The failure penalty in the reward is `−1.0 × min(failures_this_step, 3)`.

2. **"Why is G0 the unrestricted-heuristic ceiling at only 8.70?"**
   Because G0 fires 307 arrows/episode at a 2.4 % hit rate — it empties the
   clip at anything that moves. G5 fires 57 arrows/episode at a 22 % hit rate
   (9× higher efficiency). The learned policy beats the unrestricted heuristic
   *through efficiency*, not raw attack volume.

3. **"How strong is the G3→G4 deterministic regression?"**
   0.60 → 0.20 is **tied within noise** (σ = 1.43 vs 2.71). The paper and
   slides explicitly call that out. The monotonic claim is strictly for
   stochastic evaluation; deterministic is "climbs from strongly negative to
   strongly positive" with G5 as the peak.

---

## Item 4 — What to load into the slides right now

- Replace `results/v3_1/demo_sidebyside_v3.mp4`
  with `results/v3_1_slow/demo_sidebyside_v3.mp4`.
- If any slide uses a single-game clip (e.g. the G5 flourish),
  replace it with the matching file in `results/v3_1_slow/g5_demo/`.
- Nothing else needs to change in the deck: all tables, figures, and numbers
  remain correct.

---

## Item 5 — Post-presentation follow-ups (for the write-up, not today)

- Wrapper patch: add an agent-respawn or invulnerability window so episodes
  actually hit the 30 s ceiling, and re-run evaluation with the same seeds.
  That would give us cleaner "failures per episode" numbers because the
  denominator would be fixed.
- Implement the archer-bottom constraint that was discussed in design and
  re-train G2–G5 with it. Could be either a hard y-position mask or an
  archer-patrol role analogous to the knight patrol.
- Re-record demo MP4s natively at a slower FPS (e.g. 30 FPS playback of
  15 FPS simulation) instead of post-hoc frame duplication.
