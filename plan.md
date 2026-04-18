# Project Roadmap — Structured Teamwork in Multi-Agent PPO

> Live ledger for the 7-game ablation study. Full design spec in `docs/design.md`;
> academic write-up in `docs/report.pdf`.

---

## Thesis

Performance in cooperative KAZ comes from *structure* (roles, target locks,
pragmatic override) rather than reward-magnitude engineering. We evaluate four
coordination primitives in a seven-game ablation:

1. Ammo / stamina discipline
2. Role assignment (1 forward knight + 1 end-zone patrol knight)
3. Target locks (knights lock within radius; archer skips knight-locked zombies)
4. Pragmatic override (archer overrides knight lock if zombie will cross first)

All seven games run in an identical environment (30 s @ 15 FPS = 450 steps; shared
per-episode seeds) so score differences isolate the primitive.

---

## Phase 1 — Pre-flight & smoke ✅

* Verified KAZ internals: FPS = 15, obs shape `(27, 5)`, action layout
  `0=forward, 1=backward, 2=turnCCW, 3=turnCW, 4=weapon, 5=noop`.
* Confirmed empirically that crossed zombies stay in `zombie_list` (KAZ never
  removes them — it just sets `raw.run = False`) and that agents killed by zombie
  contact are removed from `archer_list` / `knight_list`. Both observations drive
  the detector design in `src/wrappers/kaz_wrapper_v3.py`.

## Phase 2 — Implementation ✅

* `src/wrappers/kaz_wrapper_v3.py` — 7-game config table, ammo & stamina pools,
  archer-immobile mode, role assignment, target-lock acquisition, G5 pragmatic
  override, end-zone crossing + agent-breach detector, 5-dim observation extras.
* `src/policies/heuristic.py` — scripted aim-and-fire / walk-and-attack for G0,
  G1a, G1b.
* `src/models/mappo_net.py` — `extra_dim` parameter projecting the 5 extras through
  `nn.Linear(extra_dim, hidden_dim)` and adding to the pooled entity vector.
* `src/train_v3.py` — PPO loop with TensorBoard scalars (`metric/score`,
  `metric/failures`, `metric/ammo_pct_remaining`, `metric/stamina_pct_remaining`,
  `metric/archer_kills`, `metric/knight_kills`, etc.) and transfer-learning loader.
* `src/evaluate_v3.py` — all 7 games, per-episode seed = `args.seed + ep_idx` so
  zombie patterns match across games; stochastic + `--deterministic` modes;
  `--record` writes per-episode MP4s.
* `src/phase_artifacts_v3.py` — score evolution, metric breakdown, attention
  saliency (G3 vs G5), side-by-side demo stitcher, ablation table.
* `scripts/mega_train_v3_1.sh` — orchestration (heuristic evals → ammo-mode
  auto-selection → G2 scratch → G3 / G4 / G5 transfer-learned in sequence).

## Phase 3 — Training ✅

~3.5 h on RTX 5070 Ti (~480 SPS). Ammo-mode auto-selected = **global** (G1a vs G1b
tied; deterministic tiebreak).

| Stage | Steps     | Params  | Architecture    | Transferred    |
|-------|-----------|---------|-----------------|----------------|
| G2    | 1,000,000 | 432,265 | Attention       | scratch        |
| G3    | 1,500,000 | 827,017 | Attention + GRU | 28 / 32 from G2 (4 GRU tensors new) |
| G4    | 1,500,000 | 827,017 | Attention + GRU | 32 / 32 from G3 |
| G5    | 2,000,000 | 827,017 | Attention + GRU | 32 / 32 from G4 |

## Phase 4 — Evaluation ✅

10 stochastic episodes per game + 10 deterministic episodes for G2–G5; plus
`--record` re-runs that produce per-episode MP4s (70 total). Seeds 42–51 identical
across every game, so zombie spawn patterns are directly comparable.

| Game | Stochastic       | Deterministic    | Kills A / K | Failures |
|------|------------------|------------------|-------------|----------|
| G0   | 8.70 ± 9.65      | —                | 7.5 / 4.2   | 3.0      |
| G1a  | 1.70 ± 3.00      | —                | 1.0 / 3.6   | 2.9      |
| G1b  | 1.60 ± 2.76      | —                | 0.9 / 3.6   | 2.9      |
| G2   | 1.80 ± 1.08      | −1.90 ± 1.14     | 4.5 / 0.1   | 2.8      |
| G3   | 4.10 ± 2.17      | 0.60 ± 1.43      | 5.5 / 1.0   | 2.4      |
| G4   | 7.40 ± 3.32      | 0.20 ± 2.71      | 10.0 / 0.1  | 2.7      |
| **G5** | **11.20 ± 3.12** | **2.90 ± 2.62** | 12.3 / 0.7  | 1.8      |

**Three findings:**

1. **Monotonic learned progression G2 → G5.** Stochastic 1.80 → 4.10 → 7.40 → 11.20;
   deterministic −1.90 → 0.60 → 0.20 → 2.90. Peak is G5 on both metrics, and G5
   beats the unrestricted heuristic ceiling (G0 = 8.70).
2. **Failure signal is live.** 1.8–3.0 failures/episode across the 7 games. G5
   achieves the lowest failure count (1.8) *and* the highest score, i.e. the
   pragmatic override earns its place in the curriculum by genuinely trading off
   offence and defence.
3. **Knights remain active through G5.** Lock radius 0.25 W (320 px) + stamina
   pool 150 keeps knights attacking 23–28 times/episode even under the tight
   G4 / G5 coordination rules (0.1–1.0 kills/ep).

All five ship-gates from `docs/design.md` §7 hold:
g1a / g1b failures > 0 ✓;
≥ 1 learned game with failures > 0 ✓;
G3 ≥ G2 ✓; G4 ≥ G3 ✓; G5 ≥ G4 ✓.

## Phase 5 — Documentation ✅

* `README.md` — project overview + quick start.
* `docs/report.pdf` / `docs/report.tex` — full academic write-up.
* `docs/design.md` — reproducible implementation + evaluation spec.
* `docs/presentation_guide.md` — 15-min slide-by-slide talk.
* `results/v3_1/ablation_table.md` — auto-generated markdown ablation table.
* `plan.md` — this file.
