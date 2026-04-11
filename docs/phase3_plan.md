# Phase 3 Plan — The Altruistic Hero (Game 4: 60/40 Comrade Healthcare)

## Context & Motivation

### Where We Stand After Phase 2
| Game | Raw Kill Density | Mean Return | Deterministic Shoot % | Narrative |
|------|-----------------|-------------|----------------------|-----------|
| G1 | 0.01089 | 5.80 ± 2.94 | 39.9% | Greedy Soldier — kills freely |
| G2 | 0.00431 | 0.30 ± 0.39 | 0.15% | Risk Avoider — won't shoot |
| G3 | 0.00279 | 0.15 ± 0.24 | 0.0% | Fully Passive — won't act at all |

**The problem:** Penalties dominate the sparse kill reward, so agents learn avoidance instead of efficiency. Each constraint layer makes agents MORE passive, not MORE strategic.

### Why Game 4 Is the Turning Point
Game 4 adds **60/40 comrade healthcare** — the first *cooperative* signal in the ablation.

**The math that matters:**
- When a teammate gets a kill (+1 raw reward), the agent receives `0.4 × mean(teammate_raw_kills)` as preservation reward
- With 3 teammates, one teammate kill → ~0.133 bonus to each other agent
- The knight's stamina cost per move is only 0.01 — **the team bonus from enabling one archer kill (0.133) outweighs 13 movement steps**
- The archer's dry-fire penalty is −0.5, but a successful kill now gives: `0.6 × 1.0 + 0.4 × 0.0 = 0.6` to self, plus 0.133 to each teammate → total team value = 0.6 + 3 × 0.133 ≈ **1.0**

**The hypothesis:** Team reward breaks the penalty avoidance loop. Agents now have a REASON to be active — protecting teammates so they can generate positive rewards that flow back.

### Expected Emergent Behavior: "The Guardian"
- **Knight:** Moves to intercept zombies approaching archers (0.01 stamina cost << 0.133 team reward per archer kill). Actively creates safe zones.
- **Archer:** Shoots more selectively than G1 but more often than G2, because each kill is amplified by team reward flowing to knights.
- **Team:** Implicit role specialization — knights tank/intercept, archers deal damage.

---

## Pre-Flight Checks

### 1. Verify Wrapper Configuration
```bash
./.venv/bin/python -c "
from src.wrappers.kaz_wrapper import KAZWrapper
env = KAZWrapper(game_level=4, vector_state=True, seed=42)
print(f'Game level:        {env.game_level}')
print(f'Ammo limit:        {env.ammo_limit_enabled}')
print(f'Stamina:           {env.stamina_enabled}')
print(f'Team reward:       {env.team_reward_enabled}')
print(f'Fog:               {env.fog_enabled}')
print(f'Weight self/team:  {env.weight_self}/{env.weight_team}')
print(f'Dry fire penalty:  {env.dry_fire_penalty}')
print(f'Stamina move/atk:  {env.stamina_cost_move}/{env.stamina_cost_attack}')
env.close()
"
```

**Expected output:**
```
Game level:        4
Ammo limit:        True
Stamina:           True
Team reward:       True
Fog:               False
Weight self/team:  0.6/0.4
Dry fire penalty:  -0.5
Stamina move/atk:  0.01/0.05
```

### 2. Verify Team Reward Flows Correctly
```bash
./.venv/bin/python -c "
import numpy as np
from src.wrappers.kaz_wrapper import KAZWrapper

env = KAZWrapper(game_level=4, vector_state=True, seed=42, max_cycles=200)
obs, _ = env.reset()

np.random.seed(42)
pres_seen = False
for step in range(100):
    if not env.agents:
        break
    actions = {a: int(np.random.randint(0, 6)) for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    for agent in rewards:
        ri = infos[agent]['reward_info']
        if ri['preservation'] != 0.0:
            print(f'Step {step}: {agent} got preservation={ri[\"preservation\"]:.3f} (raw_kill={ri[\"raw_kill\"]:.1f}, aggr={ri[\"aggression\"]:.3f}, total={ri[\"total\"]:.3f})')
            pres_seen = True
if not pres_seen:
    print('WARNING: No preservation reward observed in 100 steps — team reward may not be flowing')
env.close()
"
```

### 3. Smoke Test (Short Training Run)
```bash
./.venv/bin/python src/train.py \
    --game_level 4 \
    --total_timesteps 5000 \
    --rollout_steps 256 \
    --log_interval 1 \
    --save_interval 100 \
    --max_cycles 100
```

**Verify in output:**
- `pres` values appear in the log lines (not always zero)
- No crashes or NaN losses
- TensorBoard directory created at `results/tensorboard/game4/`

---

## Training

### Full Training Command
```bash
./.venv/bin/python src/train.py \
    --game_level 4 \
    --total_timesteps 500000 \
    --max_cycles 900
```

**Hyperparameters:** Same as G1-G3 defaults (lr=3e-4, gamma=0.99, rollout=2048, etc.). The team reward changes the optimization landscape without requiring hyperparameter surgery.

**What to watch in TensorBoard during training:**
```bash
./.venv/bin/python -m tensorboard.main --logdir results/tensorboard/ --bind_all
```

Key signals:
- `reward/preservation` should become non-zero and grow — proves team coordination is emerging
- `reward/raw_kills` should recover from G3's near-zero levels
- `metrics/episode_length` should increase (longer survival = active defense working)
- `metrics/raw_kill_density` is the north-star metric — should recover toward G1 levels
- `policy/entropy` should stabilize above G3's collapsed value (agents exploring diverse strategies)

**Estimated time:** ~15 min on RTX 5070 Ti (500K steps).

---

## Evaluation

### 1. Stochastic Evaluation (10 episodes)
```bash
./.venv/bin/python src/evaluate.py \
    --checkpoint models/game4/final.pt \
    --episodes 10
```

### 2. Deterministic Evaluation (10 episodes)
```bash
./.venv/bin/python src/evaluate.py \
    --checkpoint models/game4/final.pt \
    --episodes 10 \
    --deterministic
```

**Critical comparison:** If deterministic eval also shows kills (unlike G2/G3's zeros), it confirms the team reward broke the avoidance loop — the policy INTENTIONALLY fights, not just accidentally.

### 3. Action Distribution Analysis
```bash
./.venv/bin/python -c "
import torch
import numpy as np
from src.models.mappo_net import MAPPONet
from src.wrappers.kaz_wrapper import KAZWrapper

device = torch.device('cuda')
ckpt = torch.load('models/game4/final.pt', map_location=device, weights_only=False)
ta = ckpt['args']

env = KAZWrapper(game_level=4, vector_state=True, seed=123)
obs, _ = env.reset()
net = MAPPONet(obs_dim=135, act_dim=6, hidden_dim=ta.get('hidden_dim', 256)).to(device)
net.load_state_dict(ckpt['model_state_dict'])
net.eval()

knight_actions = np.zeros(6)
archer_actions = np.zeros(6)
for step in range(500):
    if not env.agents:
        obs, _ = env.reset()
    actions = {}
    with torch.no_grad():
        for agent in env.agents:
            obs_t = torch.tensor(obs[agent].flatten(), dtype=torch.float32, device=device).unsqueeze(0)
            out = net.forward(obs_t)
            action = out['logits'].argmax(dim=-1).item()
            actions[agent] = action
            if agent.startswith('knight'):
                knight_actions[action] += 1
            else:
                archer_actions[action] += 1
    obs, _, _, _, _ = env.step(actions)
    obs = {a: ob for a, ob in obs.items()} if env.agents else {}

k_total = knight_actions.sum()
a_total = archer_actions.sum()
print('Action map: 0=noop 1-3=move 4=shoot/attack 5=turn')
print(f'Knight (deterministic): {knight_actions / max(k_total,1)}')
print(f'Archer (deterministic): {archer_actions / max(a_total,1)}')
print(f'Knight attack %: {knight_actions[4] / max(k_total,1) * 100:.1f}%')
print(f'Archer shoot %:  {archer_actions[4] / max(a_total,1) * 100:.1f}%')
env.close()
"
```

**Success criteria:** Knight attack % > 0% and Archer shoot % > 0.15% (G2 baseline). If both increase, team reward is working.

---

## Demo Recording

### Record 10 Episodes
```bash
./.venv/bin/python src/evaluate.py \
    --checkpoint models/game4/final.pt \
    --episodes 10 \
    --record
```

**Artifacts:** `results/game4_demo/episode_{1..10}.mp4`

---

## Comparative Analysis

### Updated Ablation Table
After evaluation, update `results/phase2_ablation_table.md` → rename to `results/ablation_table.md` (now covers G1-G4) with columns:

| Metric | G1 | G2 | G3 | G4 |
|--------|----|----|----|----|
| Mean Return | | | | |
| Raw Kills | | | | |
| Raw Kill Density | | | | |
| Preservation Reward | | | | |
| Mean Ep Length | | | | |
| Det. Shoot/Attack % | | | | |

### Key Questions to Answer in Observations
1. **Recovery arc:** Did raw kill density recover from G3's 0.00279? By how much?
2. **Preservation signal:** Is the mean preservation reward meaningfully non-zero?
3. **Survival improvement:** Did episode length increase from G3's 179? (Team coordination → longer survival)
4. **Deterministic policy health:** Does the deterministic policy show intentional offensive actions, unlike G2/G3?
5. **Role differentiation:** Do knights and archers show different action distributions? (Knights moving more, archers shooting more)

### Write Observations
Create `results/phase3_observations.md` with:
- Behavioral analysis for G4
- Comparison to G1-G3 trajectory
- Whether the "Guardian" hypothesis held
- Implications for G5 (Phase 4)

---

## Project File Updates

### Update `plan.md`
- Mark Phase 3 as ✅ Complete
- Fill in G4 training stats, behavior label, and results

### Update `README.md`
- Update Progress table with G4 results

---

## Artifact Checklist

| Artifact | Path | Status |
|----------|------|--------|
| Game 4 checkpoint | `models/game4/final.pt` | ⬚ |
| Game 4 training metrics | `results/game4_baseline_metrics.json` | ⬚ |
| Game 4 eval results | `results/game4_eval_results.json` | ⬚ |
| Game 4 demo videos (10) | `results/game4_demo/` | ⬚ |
| Game 4 TensorBoard logs | `results/tensorboard/game4/` | ⬚ |
| Ablation table (G1-G4) | `results/ablation_table.md` | ⬚ |
| Phase 3 observations | `results/phase3_observations.md` | ⬚ |
| `plan.md` updated | `plan.md` | ⬚ |
| `README.md` updated | `README.md` | ⬚ |

---

## Notes for Phase 4 (G5 — The Final Hero)

Phase 4 adds Gaussian fog on top of everything. For G5 to be "miles better":
- G4 must establish a solid cooperative baseline with meaningful kill density recovery
- G5's fog will test whether the team coordination learned in G4 is robust to perceptual noise
- If G4 shows strong guardian behavior, G5 should maintain it even under uncertainty — agents that trust their teammates' positions will outperform agents that panic under fog
- Consider: G5 may benefit from slightly longer training (750K-1M steps) since partial observability makes credit assignment harder
