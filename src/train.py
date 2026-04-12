"""
MAPPO Training Loop for Knights-Archers-Zombies.

Usage:
    ./.venv/bin/python src/train.py --game_level 1 --total_timesteps 500000

Implements PPO with GAE for all agents sharing one network (parameter sharing).
Logs reward decomposition (Aggression vs. Preservation) to TensorBoard per rules.md #9.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, ".")
from src.models.mappo_net import MAPPONet
from src.wrappers.kaz_wrapper import KAZWrapper


# ── Hyperparameter defaults ──────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MAPPO Training on KAZ")
    p.add_argument("--game_level", type=int, default=1)
    p.add_argument("--total_timesteps", type=int, default=500_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--n_epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--rollout_steps", type=int, default=2048,
                    help="Steps to collect per rollout before updating")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--arch", type=str, default="mlp",
                    choices=["mlp", "attention", "attention_gru"],
                    help="Network architecture: mlp (G1-G3), attention (G4), attention_gru (G5)")
    p.add_argument("--transfer_from", type=str, default=None,
                    help="Path to checkpoint to transfer weights from (e.g. G4 -> G5)")
    p.add_argument("--min_entropy", type=float, default=None,
                    help="Entropy floor: double ent_coef dynamically when entropy drops below this")
    p.add_argument("--num_entities", type=int, default=27,
                    help="Number of entity slots in observation (for attention arch)")
    p.add_argument("--entity_dim", type=int, default=5,
                    help="Features per entity slot (for attention arch)")
    p.add_argument("--num_heads", type=int, default=4,
                    help="Number of attention heads (for attention arch)")
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=5,
                    help="Log to TensorBoard every N rollout updates")
    p.add_argument("--save_interval", type=int, default=20,
                    help="Save checkpoint every N rollout updates")
    p.add_argument("--max_cycles", type=int, default=900)
    return p.parse_args()


# ── Rollout buffer ───────────────────────────────────────────────────
class RolloutBuffer:
    """Flat buffer for all agents (parameter sharing), with per-agent GAE."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.agent_ids = []
        # Reward decomposition tracking
        self.raw_kill_rewards = []
        self.aggression_rewards = []
        self.preservation_rewards = []
        # GRU hidden states (stored as numpy arrays, None for non-GRU arches)
        self.hidden_states = []

    def clear(self):
        self.__init__()

    def add(self, agent_id, obs, action, log_prob, reward, done, value,
            raw_kill=0.0, aggression=0.0, preservation=0.0, hidden=None):
        self.agent_ids.append(agent_id)
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.raw_kill_rewards.append(raw_kill)
        self.aggression_rewards.append(aggression)
        self.preservation_rewards.append(preservation)
        self.hidden_states.append(hidden)

    def compute_gae(self, last_values: dict, gamma: float, gae_lambda: float):
        """Compute GAE per-agent to avoid cross-agent contamination.

        The flat buffer interleaves transitions from multiple agents.
        Walking backwards across the flat buffer would incorrectly mix
        value estimates between different agents.  Instead, we group
        buffer indices by agent_id and compute GAE within each group.
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)

        # Group flat-buffer indices by agent
        agent_indices = defaultdict(list)
        for i, aid in enumerate(self.agent_ids):
            agent_indices[aid].append(i)

        for aid, indices in agent_indices.items():
            last_val = last_values.get(aid, 0.0)
            m = len(indices)
            last_gae = 0.0

            for k in reversed(range(m)):
                idx = indices[k]
                if k == m - 1:
                    next_value = last_val
                else:
                    next_idx = indices[k + 1]
                    next_value = self.values[next_idx]

                next_non_terminal = 1.0 - float(self.dones[idx])
                delta = (self.rewards[idx]
                         + gamma * next_value * next_non_terminal
                         - self.values[idx])
                advantages[idx] = last_gae = (
                    delta + gamma * gae_lambda * next_non_terminal * last_gae
                )

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def to_tensors(self, device):
        result = {
            "obs": torch.tensor(np.array(self.obs), dtype=torch.float32, device=device),
            "actions": torch.tensor(np.array(self.actions), dtype=torch.long, device=device),
            "log_probs": torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device),
        }
        if self.hidden_states and self.hidden_states[0] is not None:
            result["hidden_states"] = torch.tensor(
                np.array(self.hidden_states), dtype=torch.float32, device=device
            )
        else:
            result["hidden_states"] = None
        return result


# ── Training ─────────────────────────────────────────────────────────
def flatten_obs(obs_dict):
    """Flatten (N, M) vector obs to (N*M,) for each agent."""
    return {a: ob.flatten().astype(np.float32) for a, ob in obs_dict.items()}


def train(args):
    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    print(f"[train] Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"[train] Game level: {args.game_level}, arch: {args.arch}")

    # Environment
    env = KAZWrapper(
        game_level=args.game_level,
        max_cycles=args.max_cycles,
        vector_state=True,
        seed=args.seed,
    )
    obs_raw, _ = env.reset()
    obs_flat = flatten_obs(obs_raw)
    obs_dim = list(obs_flat.values())[0].shape[0]  # 135
    act_dim = env.action_space(env.agents[0]).n      # 6
    print(f"[train] obs_dim={obs_dim}, act_dim={act_dim}, agents={env.possible_agents}")

    # Network + optimizer
    net = MAPPONet(
        obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim,
        arch=args.arch, num_entities=args.num_entities,
        entity_dim=args.entity_dim, num_heads=args.num_heads,
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    print(f"[train] Network params: {sum(p.numel() for p in net.parameters()):,}")

    # Transfer weights from a prior checkpoint (e.g. G4 attention -> G5 attention_gru)
    if args.transfer_from:
        g_ckpt = torch.load(args.transfer_from, map_location=device, weights_only=False)
        g_state = g_ckpt["model_state_dict"]
        g5_state = net.state_dict()
        transferred, skipped = 0, 0
        for key in g_state:
            if key in g5_state and g_state[key].shape == g5_state[key].shape:
                g5_state[key] = g_state[key]
                transferred += 1
            else:
                skipped += 1
        net.load_state_dict(g5_state)
        print(f"[train] Transferred {transferred} param tensors from {args.transfer_from} "
              f"(skipped {skipped})")

    # TensorBoard (rules.md #9)
    log_dir = f"results/tensorboard/game{args.game_level}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Checkpoint dir
    ckpt_dir = f"models/game{args.game_level}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Main training loop ───────────────────────────────────────
    global_step = 0
    update_count = 0
    episode_count = 0
    episode_returns = []       # track last N episode returns
    episode_lengths = []

    # Per-agent GRU hidden states (None for non-GRU arches)
    # Keyed by agent name; reset on episode boundaries
    agent_hidden = {}

    # Running episode trackers
    ep_return = defaultdict(float)
    ep_raw_kills = defaultdict(float)
    ep_aggression = defaultdict(float)
    ep_preservation = defaultdict(float)
    ep_length = 0

    obs_raw, _ = env.reset()
    obs_flat = flatten_obs(obs_raw)
    start_time = time.time()

    while global_step < args.total_timesteps:
        buffer = RolloutBuffer()
        rollout_ep_returns = []
        rollout_ep_lengths = []
        rollout_raw_kills = []
        rollout_aggression = []
        rollout_preservation = []

        # ── Collect rollout ──────────────────────────────────
        for _ in range(args.rollout_steps):
            if not env.agents:
                # Episode ended — record metrics and reset
                mean_return = float(np.mean(list(ep_return.values())))
                episode_returns.append(mean_return)
                episode_lengths.append(ep_length)
                rollout_ep_returns.append(mean_return)
                rollout_ep_lengths.append(ep_length)
                rollout_raw_kills.append(
                    float(np.mean(list(ep_raw_kills.values()))) if ep_raw_kills else 0.0
                )
                rollout_aggression.append(
                    float(np.mean(list(ep_aggression.values()))) if ep_aggression else 0.0
                )
                rollout_preservation.append(
                    float(np.mean(list(ep_preservation.values()))) if ep_preservation else 0.0
                )
                episode_count += 1
                ep_return = defaultdict(float)
                ep_raw_kills = defaultdict(float)
                ep_aggression = defaultdict(float)
                ep_preservation = defaultdict(float)
                ep_length = 0
                obs_raw, _ = env.reset()
                obs_flat = flatten_obs(obs_raw)
                agent_hidden = {}  # reset GRU states on episode boundary

            # Get actions for all living agents
            with torch.no_grad():
                for agent in env.agents:
                    obs_t = torch.tensor(
                        obs_flat[agent], dtype=torch.float32, device=device
                    ).unsqueeze(0)

                    # Retrieve (or initialise) this agent's GRU hidden state
                    h_np = agent_hidden.get(agent)  # shape (hidden_dim,) or None
                    if h_np is not None:
                        h_t = torch.tensor(
                            h_np, dtype=torch.float32, device=device
                        ).unsqueeze(0)  # (1, hidden_dim)
                    else:
                        h_t = None

                    action, log_prob, _, value, _, new_hidden = net.get_action_and_value(
                        obs_t, hidden=h_t
                    )

                    # Persist updated hidden state (detached)
                    if new_hidden is not None:
                        agent_hidden[agent] = (
                            new_hidden.detach().cpu().numpy()[0]  # (hidden_dim,)
                        )

                    # For GRU arch: store zeros when agent has no prior hidden state
                    # so that the buffer's hidden_states list stays homogeneous.
                    if args.arch == "attention_gru":
                        h_store = (
                            h_np if h_np is not None
                            else np.zeros(args.hidden_dim, dtype=np.float32)
                        )
                    else:
                        h_store = None

                    buffer.add(
                        agent_id=agent,
                        obs=obs_flat[agent],
                        action=action.item(),
                        log_prob=log_prob.item(),
                        reward=0.0,   # filled after step
                        done=False,
                        value=value.item(),
                        hidden=h_store,
                    )

            # Step env
            actions = {}
            buf_len = len(buffer.actions)
            agent_indices = {}
            idx = buf_len - len(env.agents)
            for agent in env.agents:
                actions[agent] = buffer.actions[idx]
                agent_indices[agent] = idx
                idx += 1

            obs_raw, rewards, terms, truncs, infos = env.step(actions)
            obs_flat = flatten_obs(obs_raw) if env.agents else {}

            for agent in rewards:
                ri = infos.get(agent, {}).get("reward_info", {})
                raw_k = ri.get("raw_kill", 0.0)
                aggr = ri.get("aggression", 0.0)
                pres = ri.get("preservation", 0.0)
                total = ri.get("total", rewards[agent])
                done = terms.get(agent, False) or truncs.get(agent, False)

                buf_idx = agent_indices[agent]
                buffer.rewards[buf_idx] = total
                buffer.dones[buf_idx] = done
                buffer.raw_kill_rewards[buf_idx] = raw_k
                buffer.aggression_rewards[buf_idx] = aggr
                buffer.preservation_rewards[buf_idx] = pres

                ep_return[agent] += total
                ep_raw_kills[agent] += raw_k
                ep_aggression[agent] += aggr
                ep_preservation[agent] += pres

            ep_length += 1
            global_step += len(rewards)

        # ── Compute GAE (per-agent) ─────────────────────────────
        last_values = {}
        if env.agents:
            with torch.no_grad():
                for agent in env.agents:
                    obs_t = torch.tensor(
                        obs_flat[agent], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    h_np = agent_hidden.get(agent)
                    h_t = (
                        torch.tensor(h_np, dtype=torch.float32, device=device).unsqueeze(0)
                        if h_np is not None else None
                    )
                    _, _, _, v, _, _ = net.get_action_and_value(obs_t, hidden=h_t)
                    last_values[agent] = v.item()

        advantages, returns = buffer.compute_gae(last_values, args.gamma, args.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        tensors = buffer.to_tensors(device)
        all_obs = tensors["obs"]
        all_actions = tensors["actions"]
        all_old_log_probs = tensors["log_probs"]
        all_hidden = tensors["hidden_states"]  # (N, hidden_dim) or None
        all_returns = torch.tensor(returns, dtype=torch.float32, device=device)
        all_advantages = torch.tensor(advantages, dtype=torch.float32, device=device)

        # ── PPO Update ───────────────────────────────────────
        n_samples = len(all_obs)
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []

        for epoch in range(args.n_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, args.batch_size):
                end = min(start + args.batch_size, n_samples)
                mb_idx = indices[start:end]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=device)

                mb_obs = all_obs[mb_idx_t]
                mb_actions = all_actions[mb_idx_t]
                mb_old_lp = all_old_log_probs[mb_idx_t]
                mb_returns = all_returns[mb_idx_t]
                mb_advantages = all_advantages[mb_idx_t]
                mb_hidden = (
                    all_hidden[mb_idx_t].detach() if all_hidden is not None else None
                )

                _, new_lp, entropy, new_val, _, _ = net.get_action_and_value(
                    mb_obs, mb_actions, mb_hidden
                )

                # Policy loss (clipped)
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - args.clip_range,
                                    1.0 + args.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_val, mb_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Entropy floor: double ent_coef if entropy collapses below threshold
                eff_ent_coef = args.ent_coef
                if args.min_entropy is not None and entropy.mean().item() < args.min_entropy:
                    eff_ent_coef = args.ent_coef * 2.0

                loss = (policy_loss
                        + args.vf_coef * value_loss
                        + eff_ent_coef * entropy_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())
                total_losses.append(loss.item())

        update_count += 1

        # ── Logging ──────────────────────────────────────────
        if update_count % args.log_interval == 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed

            avg_policy_loss = float(np.mean(policy_losses))
            avg_value_loss = float(np.mean(value_losses))
            avg_entropy = float(np.mean(entropy_losses))

            # Reward decomposition (per rules.md #9)
            avg_raw_kills = float(np.mean(buffer.raw_kill_rewards))
            avg_aggression = float(np.mean(buffer.aggression_rewards))
            avg_preservation = float(np.mean(buffer.preservation_rewards))
            avg_reward = float(np.mean(buffer.rewards))

            writer.add_scalar("reward/total", avg_reward, global_step)
            writer.add_scalar("reward/raw_kills", avg_raw_kills, global_step)
            writer.add_scalar("reward/aggression", avg_aggression, global_step)
            writer.add_scalar("reward/preservation", avg_preservation, global_step)
            writer.add_scalar("policy/loss", avg_policy_loss, global_step)
            writer.add_scalar("policy/entropy", avg_entropy, global_step)
            writer.add_scalar("value/loss", avg_value_loss, global_step)
            writer.add_scalar("training/sps", sps, global_step)

            if rollout_ep_returns:
                mean_ep_return = float(np.mean(rollout_ep_returns))
                mean_ep_length = float(np.mean(rollout_ep_lengths))
                mean_raw_kills = float(np.mean(rollout_raw_kills))
                mean_aggr = float(np.mean(rollout_aggression))
                mean_pres = float(np.mean(rollout_preservation))
                writer.add_scalar("metrics/episode_return", mean_ep_return, global_step)
                writer.add_scalar("metrics/episode_length", mean_ep_length, global_step)
                writer.add_scalar("metrics/raw_kill_density",
                                  mean_raw_kills / max(mean_ep_length, 1), global_step)
                writer.add_scalar("metrics/kill_density",
                                  mean_aggr / max(mean_ep_length, 1), global_step)
                writer.add_scalar("metrics/survival_time", mean_ep_length, global_step)

            print(
                f"[update {update_count:>4d}] step={global_step:>8d}  "
                f"ep={episode_count:>4d}  "
                f"R={avg_reward:+.4f}  "
                f"aggr={avg_aggression:+.4f}  pres={avg_preservation:+.4f}  "
                f"pi_loss={avg_policy_loss:.4f}  v_loss={avg_value_loss:.4f}  "
                f"ent={avg_entropy:.4f}  SPS={sps:.0f}"
            )

        # ── Save checkpoint ──────────────────────────────────
        if update_count % args.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{global_step}.pt")
            torch.save({
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "update_count": update_count,
                "args": vars(args),
            }, ckpt_path)
            print(f"[checkpoint] Saved to {ckpt_path}")

    # ── Final save ───────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "final.pt")
    torch.save({
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "update_count": update_count,
        "args": vars(args),
    }, final_path)
    print(f"[final] Model saved to {final_path}")

    # Save baseline metrics
    metrics = {
        "game_level": args.game_level,
        "total_timesteps": global_step,
        "episodes": episode_count,
        "mean_episode_return": float(np.mean(episode_returns[-50:])) if episode_returns else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths[-50:])) if episode_lengths else 0.0,
        "mean_kill_density": (
            float(np.mean(episode_returns[-50:])) / max(float(np.mean(episode_lengths[-50:])), 1)
            if episode_returns else 0.0
        ),
    }
    metrics_path = f"results/game{args.game_level}_baseline_metrics.json"
    os.makedirs("results", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics] Saved to {metrics_path}")

    writer.close()
    env.close()
    print("[train] Done.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
