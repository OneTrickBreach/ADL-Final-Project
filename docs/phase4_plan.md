# Phase 4 Plan — The Final Hero (Game 5: Gaussian Fog + GRU Recurrence)

## Context & Motivation

### Where We Stand After Phase 3

| Game | Raw Kill Density | Mean Return | Det. Kills | Stoch. Shoot % | Narrative |
|------|-----------------|-------------|------------|----------------|-----------|
| G1 | 0.01089 | 5.80 ± 2.94 | Yes (1.23) | 39.9% | Greedy Soldier |
| G2 | 0.00431 | 0.30 ± 0.39 | 0 | 0.15% | Risk Avoider |
| G3 | 0.00279 | 0.15 ± 0.24 | 0 | 0.0% | Fully Passive |
| G4 | 0.00361 | 0.53 ± 0.35 | 0 | 8.6% | Recovering Cooperator |

**G4 reversed the kill decline** (+29% density from G3) and proved team reward works (preservation = +0.68). But **the deterministic policy is still passive** — 0 kills under argmax in G2/G3/G4.

### The Two Problems G5 Must Solve

1. **Fog:** Gaussian noise on observations degrades entity position information. Agents can't reliably determine zombie locations, making precise attacks harder.
2. **Deterministic passivity:** The policy mode (argmax) has been passive since G2. If G5 doesn't fix this, the ablation story becomes "constraints always win" — valid but not the hero narrative.

### Why GRU Recurrence Fixes Both

**For fog:** A GRU hidden state accumulates evidence over time. Even when a single fogged observation is noisy, the running hidden state integrates multiple observations, acting as a **Bayesian belief update** — "I saw a zombie near position X on step t-3, and movement noise is low, so it's probably still near X."

**For deterministic passivity:** The GRU provides *temporal context* that the feedforward attention encoder lacks. An agent can now distinguish:
- "A zombie just appeared far away" (no action needed) vs.
- "A zombie has been approaching for 5 steps and is now close" (attack NOW)

This state-conditional confidence is what's needed for the argmax to shift to offensive in specific states, rather than always defaulting to the safest action.

### Expected Emergent Behavior: "Fire Discipline"
- **Knight:** Tracks zombie approach trajectories through fog. Commits to interception only when the hidden state confirms proximity. Protects archers based on remembered teammate positions.
- **Archer:** Accumulates evidence of zombie location before shooting. Fewer wasted shots than G1 (resource aware), but more confident shots than G4 (temporal certainty).
- **Team:** Role specialization becomes temporal — knights act as early-warning interceptors, archers as confirmed-kill finishers.

---

## Architecture: Attention + GRU

### New `--arch attention_gru` Mode

```
Obs (batch, 135) → reshape (batch, 27, 5)
  → Linear(5, 256) per entity              # entity embedding
  → MultiHeadSelfAttention(256, 4 heads)   # cross-entity reasoning
  → Residual + LayerNorm
  → MaskedMeanPool over active entities     # (batch, 256)
  → GRUCell(256, 256)                       # temporal memory    ← NEW
  → Linear(256, 256) + ReLU                 # backbone
  → Policy Head / Value Head
```

**Key design decisions:**
- **GRUCell, not GRU:** We process one timestep at a time during rollout (not full sequences), so GRUCell is the right API.
- **Hidden state management:** Per-agent hidden states maintained during rollout and passed to the buffer (detached from compute graph). Reset on episode boundaries and agent death.
- **PPO update:** Stored hidden states used as context input. Gradients do NOT flow through the hidden state (detached) — this is the standard approach for PPO + recurrence.
- **Return signature change:** `get_action_and_value` returns 6 values (adding `new_hidden`). For MLP/attention, the 6th value is `None`.

### Implementation Changes Required

**`mappo_net.py`:**
- Add `"attention_gru"` to `ARCH_CHOICES`
- Add `self.gru = nn.GRUCell(hidden_dim, hidden_dim)` when arch is `"attention_gru"`
- Update `forward()` to accept optional `hidden` param, return `new_hidden` in output dict
- Update `get_action_and_value()` to pass through hidden state and return 6 values

**`train.py`:**
- Add `"attention_gru"` to `--arch` choices
- Maintain `agent_hidden_states: dict[str, torch.Tensor]` during rollout
- Store detached hidden states in `RolloutBuffer`
- Pass hidden states to `get_action_and_value()` during rollout and PPO update
- Reset hidden states on episode boundaries and agent death

**`evaluate.py`:**
- Maintain per-agent hidden states during evaluation
- Reset on episode start

### Transfer Learning from G4 (Recommended)

Instead of training from scratch, **load G4's attention weights** into G5's `attention_gru` model:

```python
# Load G4 checkpoint
g4_ckpt = torch.load('models/game4/final.pt', ...)
g4_state = g4_ckpt['model_state_dict']

# Create G5 model (attention_gru)
net = MAPPONet(obs_dim=135, act_dim=6, arch='attention_gru', ...)

# Partial load: copy matching keys (entity_attention.*, policy_head.*, value_head.*, etc.)
# GRU and backbone will be randomly initialized
g5_state = net.state_dict()
transferred = 0
for key in g4_state:
    if key in g5_state and g4_state[key].shape == g5_state[key].shape:
        g5_state[key] = g4_state[key]
        transferred += 1
net.load_state_dict(g5_state)
print(f"Transferred {transferred} parameter tensors from G4")
```

**Why this helps:**
- Entity attention already learned meaningful entity relationships in G4
- Policy/value heads already have useful weights
- Only the GRU needs to be learned from scratch — much faster convergence
- The backbone (Linear+ReLU after GRU) reinitializes since input distribution changes

Add `--transfer_from` flag to train.py for this.

---

## Addressing Deterministic Passivity

### Root Cause Analysis
The argmax action is passive because:
1. Penalties are **certain** (every move costs stamina, every dry-fire costs -0.5)
2. Kills are **rare** (low hit probability per attempt)
3. The policy averages over all states → mode = safest action overall

### Strategy: State-Conditional Confidence via Memory
The GRU changes the equation: with temporal memory, the agent can recognize **high-kill-probability states** (zombie confirmed close, in line of fire, has ammo). In these states, the expected value of attacking exceeds the penalty, and the argmax should shift.

### Training Adjustments to Support This
1. **Longer training:** 750,000 steps (50% more than G1-G4) — GRU needs more data to learn temporal patterns
2. **Entropy floor:** Add `--min_entropy 0.3` flag. If entropy drops below this, scale up `ent_coef` dynamically. This prevents premature mode collapse while allowing state-conditional sharpening.
3. **Transfer from G4:** Pre-trained attention reduces the learning burden, letting the GRU focus on temporal integration

### Success Criteria
- **Primary:** Deterministic eval raw kills > 0 (breaks the G2/G3/G4 streak of zero)
- **Secondary:** Deterministic kill density > G4 stochastic (0.00361)
- **Stretch:** Stochastic kill density approaches G1 (0.01089)

---

## Pre-Flight Checks

### 1. Verify Wrapper Configuration for G5
```bash
./.venv/bin/python -c "
from src.wrappers.kaz_wrapper import KAZWrapper
env = KAZWrapper(game_level=5, vector_state=True, seed=42)
print(f'Game level:        {env.game_level}')
print(f'Ammo limit:        {env.ammo_limit_enabled}')
print(f'Stamina:           {env.stamina_enabled}')
print(f'Team reward:       {env.team_reward_enabled}')
print(f'Fog:               {env.fog_enabled}')
print(f'Fog sigma:         {env.fog_sigma}')
print(f'Weight self/team:  {env.weight_self}/{env.weight_team}')
env.close()
"
```

**Expected output:**
```
Game level:        5
Ammo limit:        True
Stamina:           True
Team reward:       True
Fog:               True
Fog sigma:         3.0
Weight self/team:  0.6/0.4
```

### 2. Verify Fog Effect on Observations
```bash
./.venv/bin/python -c "
import numpy as np
from src.wrappers.kaz_wrapper import KAZWrapper

env_clean = KAZWrapper(game_level=4, vector_state=True, seed=42)
env_fog = KAZWrapper(game_level=5, vector_state=True, seed=42)

obs_clean, _ = env_clean.reset()
obs_fog, _ = env_fog.reset()

for agent in sorted(obs_clean.keys()):
    c = obs_clean[agent].flatten()
    f = obs_fog[agent].flatten()
    diff = np.abs(c - f)
    print(f'{agent}: mean_diff={diff.mean():.4f} max_diff={diff.max():.4f} '
          f'snr={np.std(c)/max(np.std(f-c),1e-8):.2f}')

env_clean.close()
env_fog.close()
"
```

### 3. Verify GRU Architecture Builds
```bash
./.venv/bin/python -c "
import torch
from src.models.mappo_net import MAPPONet

net = MAPPONet(obs_dim=135, act_dim=6, arch='attention_gru').cuda()
print(f'Params: {sum(p.numel() for p in net.parameters()):,}')

obs = torch.randn(4, 135).cuda()
hidden = torch.zeros(4, 256).cuda()

# Test forward with hidden state
out = net(obs, hidden=hidden)
print(f'Output keys: {list(out.keys())}')
print(f'Hidden returned: {out[\"hidden\"].shape}')

# Test backward
out['value'].sum().backward()
print('Forward/backward OK')
"
```

### 4. Verify Transfer Loading from G4
```bash
./.venv/bin/python -c "
import torch
from src.models.mappo_net import MAPPONet

g4_ckpt = torch.load('models/game4/final.pt', map_location='cuda', weights_only=False)
g4_state = g4_ckpt['model_state_dict']

net = MAPPONet(obs_dim=135, act_dim=6, arch='attention_gru').cuda()
g5_state = net.state_dict()

transferred, skipped = 0, 0
for key in g4_state:
    if key in g5_state and g4_state[key].shape == g5_state[key].shape:
        g5_state[key] = g4_state[key]
        transferred += 1
    else:
        skipped += 1
        print(f'  SKIP: {key}')
net.load_state_dict(g5_state)
print(f'Transferred {transferred}, skipped {skipped}')
"
```

### 5. Smoke Test (Short Training Run)
```bash
./.venv/bin/python src/train.py \
    --game_level 5 \
    --arch attention_gru \
    --transfer_from models/game4/final.pt \
    --total_timesteps 5000 \
    --rollout_steps 256 \
    --log_interval 1 \
    --save_interval 100 \
    --max_cycles 100
```

**Verify:**
- `arch: attention_gru` printed on startup
- `Transferred N params from G4` printed
- `pres` values appear in logs
- No NaN losses
- Hidden state resets correctly on episode boundaries

---

## Training

### Full Training Command (Recommended: Transfer + Extended)
```bash
./.venv/bin/python src/train.py \
    --game_level 5 \
    --arch attention_gru \
    --transfer_from models/game4/final.pt \
    --total_timesteps 750000 \
    --max_cycles 900
```

### Alternative: From Scratch (for ablation comparison)
```bash
./.venv/bin/python src/train.py \
    --game_level 5 \
    --arch attention_gru \
    --total_timesteps 750000 \
    --max_cycles 900
```

**What to watch in TensorBoard:**
```bash
./.venv/bin/python -m tensorboard.main --logdir results/tensorboard/ --bind_all
```

Key signals:
- `reward/preservation` — should remain non-zero despite fog (team reward still works)
- `reward/raw_kills` — critical: does kill rate hold or collapse under fog?
- `policy/entropy` — should NOT drop below ~0.5 (would signal mode collapse)
- `metrics/raw_kill_density` — north star: should track toward G4 stochastic (0.00361) or higher
- `metrics/episode_length` — fog may shorten episodes initially; should recover as GRU learns

**Estimated time:** ~30 min on RTX 5070 Ti (750K steps, attention_gru slightly slower than attention).

---

## Evaluation

### 1. Stochastic Evaluation (10 episodes)
```bash
./.venv/bin/python src/evaluate.py \
    --checkpoint models/game5/final.pt \
    --episodes 10
```

### 2. Deterministic Evaluation (10 episodes) — THE KEY TEST
```bash
./.venv/bin/python src/evaluate.py \
    --checkpoint models/game5/final.pt \
    --episodes 10 \
    --deterministic
```

**This is the moment of truth.** If deterministic kills > 0, the hero narrative holds: GRU memory gives agents enough temporal confidence to commit to offensive actions under argmax.

### 3. Action Distribution Analysis
```bash
./.venv/bin/python -c "
import torch
import numpy as np
from src.models.mappo_net import MAPPONet
from src.wrappers.kaz_wrapper import KAZWrapper

device = torch.device('cuda')
ckpt = torch.load('models/game5/final.pt', map_location=device, weights_only=False)
ta = ckpt['args']
arch = ta.get('arch', 'mlp')

env = KAZWrapper(game_level=5, vector_state=True, seed=123)
obs, _ = env.reset()
net = MAPPONet(obs_dim=135, act_dim=6, hidden_dim=ta.get('hidden_dim', 256), arch=arch).to(device)
net.load_state_dict(ckpt['model_state_dict'])
net.eval()

knight_actions = np.zeros(6)
archer_actions = np.zeros(6)
hidden_states = {}
for step in range(500):
    if not env.agents:
        obs, _ = env.reset()
        hidden_states = {}
    actions = {}
    with torch.no_grad():
        for agent in env.agents:
            obs_t = torch.tensor(obs[agent].flatten(), dtype=torch.float32, device=device).unsqueeze(0)
            h = hidden_states.get(agent, None)
            if h is None:
                h = torch.zeros(1, 256, device=device)
            out = net.forward(obs_t, hidden=h)
            action = out['logits'].argmax(dim=-1).item()
            hidden_states[agent] = out.get('hidden', h)
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

**Success criteria:**
- Knight attack % > 0% (ANY offensive action under argmax)
- Archer shoot % > 0% (breaks the G2/G3/G4 deterministic zero streak)
- If both > 0%: the hero narrative holds. Attention + GRU + team reward overcame all constraints.

---

## Demo Recording

### Record 10 Episodes
```bash
./.venv/bin/python src/evaluate.py \
    --checkpoint models/game5/final.pt \
    --episodes 10 \
    --record
```

**Artifacts:** `results/game5_demo/episode_{1..10}.mp4`

---

## Comparative Analysis

### Final Ablation Table
After evaluation, create `results/final_ablation_table.md` covering G1-G5:

| Metric | G1 | G2 | G3 | G4 | G5 |
|--------|----|----|----|----|----| 
| Mean Return | | | | | |
| Raw Kills | | | | | |
| Raw Kill Density | | | | | |
| Preservation Reward | | | | | |
| Mean Ep Length | | | | | |
| Det. Mean Return | | | | | |
| Det. Shoot/Attack % | | | | | |
| Architecture | MLP | MLP | MLP | Attn | Attn+GRU |

### The Hero Narrative (for class presentation)

```
G1: MLP           + Kill reward       → Greedy Soldier       (density: 0.01089)
G2: MLP           + Ammo penalty      → Risk Avoider         (density: 0.00431, −60%)
G3: MLP           + Stamina penalty   → Fully Passive         (density: 0.00279, −74%)
G4: MLP+Attention + Team reward       → Recovering Cooperator (density: 0.00361, +29% reversal)
G5: MLP+Attn+GRU  + Fog               → Fire Discipline       (density: ???, THE HERO)
```

### Key Questions to Answer in Final Observations
1. **Hero test:** Did G5 deterministic kills break above zero?
2. **Fog resilience:** Did kill density hold despite noise? Or did fog push agents back toward passivity?
3. **GRU impact:** Compare G5 (attention_gru) vs G4 (attention) — did recurrence add value beyond attention alone?
4. **Transfer impact:** If transfer was used, how quickly did G5 converge compared to G4?
5. **Episode length:** Did GRU memory extend survival (agents track threats better)?
6. **The full arc:** Can we tell a compelling story of G1→G5 where each architectural addition (penalties → team reward → attention → GRU) builds toward disciplined cooperation?

### Write Final Observations
Create `results/phase4_observations.md` with:
- G5 behavioral analysis
- Full G1-G5 comparative trajectory
- Whether "Fire Discipline" hypothesis held
- Honest assessment: what worked, what didn't
- Implications for the class presentation

---

## Project File Updates

### Update `plan.md`
- Mark Phase 4 as ✅ Complete
- Fill in G5 results

### Update `README.md`
- Update Progress table with G5 results
- Update architecture description to include GRU

---

## Artifact Checklist

| Artifact | Path | Status |
|----------|------|--------|
| Game 5 checkpoint | `models/game5/final.pt` | ⬚ |
| Game 5 training metrics | `results/game5_baseline_metrics.json` | ⬚ |
| Game 5 eval results | `results/game5_eval_results.json` | ⬚ |
| Game 5 demo videos (10) | `results/game5_demo/` | ⬚ |
| Game 5 TensorBoard logs | `results/tensorboard/game5/` | ⬚ |
| Final ablation table (G1-G5) | `results/final_ablation_table.md` | ⬚ |
| Phase 4 observations | `results/phase4_observations.md` | ⬚ |
| `plan.md` updated | `plan.md` | ⬚ |
| `README.md` updated | `README.md` | ⬚ |

---

## If G5 Deterministic Is Still Passive (Contingency)

If the deterministic policy STILL shows 0 kills, the project narrative shifts to:

**"The Resilience of Penalty Avoidance"** — a finding that stacked penalties create such strong avoidance gradients that even advanced architectures (attention, GRU, team reward) can only improve stochastic exploration, not shift the policy mode. This is actually a publishable insight in MARL reward shaping literature.

In this case, for the presentation:
- Show the stochastic improvement arc (G3 → G4 → G5 raw kill density)
- Show the preservation reward flowing (cooperation IS happening)
- Frame the deterministic passivity as a "mode collapse under penalty dominance"
- Propose future work: reward annealing, population-based training, or intrinsic curiosity
