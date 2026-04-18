"""V3 MAPPO training loop (KAZWrapperV3).

Usage:
    ./.venv/bin/python src/train_v3.py --game g2 --ammo_mode individual --total_timesteps 1000000
    ./.venv/bin/python src/train_v3.py --game g3 --ammo_mode individual --transfer_from models/v3/g2/final.pt

Per rules.md #9, logs reward decomposition (aggression/preservation + V3 extras)
to TensorBoard.  Mirrors train.py but uses KAZWrapperV3 and extra_dim=5.
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
from src.utils import get_device, device_info
from src.wrappers.kaz_wrapper_v3 import KAZWrapperV3, EXTRA_DIM


GAME_ARCH = {"g2": "attention", "g3": "attention_gru",
             "g4": "attention_gru", "g5": "attention_gru"}


def parse_args():
    p = argparse.ArgumentParser(description="V3 MAPPO Training on KAZ")
    p.add_argument("--game", type=str, required=True, choices=list(GAME_ARCH.keys()))
    p.add_argument("--ammo_mode", type=str, default="individual",
                    choices=["global", "individual"])
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--n_epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--rollout_steps", type=int, default=2048)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_entities", type=int, default=27)
    p.add_argument("--entity_dim", type=int, default=5)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--min_entropy", type=float, default=None)
    p.add_argument("--transfer_from", type=str, default=None)
    p.add_argument("--action_mask_mode", type=str, default="soft",
                    choices=["soft", "hard"])
    p.add_argument("--duration_seconds", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=20)
    p.add_argument("--output_suffix", type=str, default="",
                   help="Appended to output dirs: models/v3<suf>/, results/v3<suf>/, "
                        "results/tensorboard_v3<suf>/. Use '_1' for V3.1 runs.")
    return p.parse_args()


class RolloutBuffer:
    def __init__(self):
        self.obs = []; self.actions = []; self.log_probs = []
        self.rewards = []; self.dones = []; self.values = []
        self.agent_ids = []
        self.raw_kill_rewards = []
        self.role_bonus_rewards = []
        self.lock_bonus_rewards = []
        self.failure_shared_rewards = []
        self.hidden_states = []

    def add(self, agent_id, obs, action, log_prob, reward, done, value,
            raw_kill=0.0, role_b=0.0, lock_b=0.0, failure_s=0.0, hidden=None):
        self.agent_ids.append(agent_id)
        self.obs.append(obs); self.actions.append(action)
        self.log_probs.append(log_prob); self.rewards.append(reward)
        self.dones.append(done); self.values.append(value)
        self.raw_kill_rewards.append(raw_kill)
        self.role_bonus_rewards.append(role_b)
        self.lock_bonus_rewards.append(lock_b)
        self.failure_shared_rewards.append(failure_s)
        self.hidden_states.append(hidden)

    def compute_gae(self, last_values, gamma, lam):
        n = len(self.rewards)
        adv = np.zeros(n, dtype=np.float32)
        agent_idx = defaultdict(list)
        for i, aid in enumerate(self.agent_ids):
            agent_idx[aid].append(i)
        for aid, idxs in agent_idx.items():
            last_val = last_values.get(aid, 0.0)
            last_gae = 0.0
            m = len(idxs)
            for k in reversed(range(m)):
                idx = idxs[k]
                if k == m - 1:
                    next_value = last_val
                else:
                    next_value = self.values[idxs[k + 1]]
                nt = 1.0 - float(self.dones[idx])
                delta = self.rewards[idx] + gamma * next_value * nt - self.values[idx]
                adv[idx] = last_gae = delta + gamma * lam * nt * last_gae
        returns = adv + np.array(self.values, dtype=np.float32)
        return adv, returns

    def to_tensors(self, device):
        r = {
            "obs": torch.tensor(np.array(self.obs), dtype=torch.float32, device=device),
            "actions": torch.tensor(np.array(self.actions), dtype=torch.long, device=device),
            "log_probs": torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device),
        }
        if self.hidden_states and self.hidden_states[0] is not None:
            r["hidden_states"] = torch.tensor(
                np.array(self.hidden_states), dtype=torch.float32, device=device)
        else:
            r["hidden_states"] = None
        return r


def flatten_obs(obs_dict):
    return {a: ob.flatten().astype(np.float32) for a, ob in obs_dict.items()}


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    arch = GAME_ARCH[args.game]
    print(f"[train_v3] device={device_info(device)} game={args.game} arch={arch} "
          f"ammo_mode={args.ammo_mode} duration={args.duration_seconds}s")

    env = KAZWrapperV3(
        game_level=args.game,
        duration_seconds=args.duration_seconds,
        ammo_mode_override=args.ammo_mode,
        action_mask_mode=args.action_mask_mode,
        vector_state=True,
        seed=args.seed,
    )
    obs_raw, _ = env.reset()
    obs_flat = flatten_obs(obs_raw)
    obs_dim = list(obs_flat.values())[0].shape[0]
    act_dim = env.action_space(env.agents[0]).n
    print(f"[train_v3] obs_dim={obs_dim} act_dim={act_dim} agents={env.possible_agents} "
          f"max_cycles={env.max_cycles}")

    net = MAPPONet(
        obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim,
        arch=arch, num_entities=args.num_entities,
        entity_dim=args.entity_dim, num_heads=args.num_heads,
        extra_dim=EXTRA_DIM,
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    print(f"[train_v3] Network params: {sum(p.numel() for p in net.parameters()):,}")

    if args.transfer_from:
        ck = torch.load(args.transfer_from, map_location=device, weights_only=False)
        src_state = ck["model_state_dict"]
        dst_state = net.state_dict()
        transferred, skipped = 0, 0
        for key in src_state:
            if key in dst_state and src_state[key].shape == dst_state[key].shape:
                dst_state[key] = src_state[key]
                transferred += 1
            else:
                skipped += 1
        net.load_state_dict(dst_state)
        print(f"[train_v3] Transferred {transferred} tensors from {args.transfer_from} "
              f"(skipped {skipped})")

    suf = args.output_suffix or ""
    log_dir = f"results/tensorboard_v3{suf}/{args.game}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    ckpt_dir = f"models/v3{suf}/{args.game}"
    os.makedirs(ckpt_dir, exist_ok=True)
    results_dir = f"results/v3{suf}"

    global_step = 0; update_count = 0; episode_count = 0
    episode_returns, episode_lengths = [], []
    agent_hidden = {}
    ep_return = defaultdict(float)
    ep_raw_kill = defaultdict(float)
    ep_role = defaultdict(float)
    ep_lock = defaultdict(float)
    ep_fail = defaultdict(float)
    ep_length = 0
    # Per-episode score (via env.get_episode_stats at boundary)
    last_ep_stats = None

    obs_raw, _ = env.reset()
    obs_flat = flatten_obs(obs_raw)
    start_time = time.time()

    while global_step < args.total_timesteps:
        buffer = RolloutBuffer()
        rollout_ep_returns = []; rollout_ep_lengths = []
        rollout_ep_stats = []

        for _ in range(args.rollout_steps):
            if not env.agents:
                mean_return = float(np.mean(list(ep_return.values()))) if ep_return else 0.0
                episode_returns.append(mean_return)
                episode_lengths.append(ep_length)
                rollout_ep_returns.append(mean_return)
                rollout_ep_lengths.append(ep_length)
                try:
                    stats = env.get_episode_stats()
                    rollout_ep_stats.append(stats)
                    last_ep_stats = stats
                except Exception:
                    pass
                episode_count += 1
                ep_return = defaultdict(float)
                ep_raw_kill = defaultdict(float)
                ep_role = defaultdict(float)
                ep_lock = defaultdict(float)
                ep_fail = defaultdict(float)
                ep_length = 0
                obs_raw, _ = env.reset()
                obs_flat = flatten_obs(obs_raw)
                agent_hidden = {}

            with torch.no_grad():
                for agent in env.agents:
                    obs_t = torch.tensor(obs_flat[agent], dtype=torch.float32,
                                         device=device).unsqueeze(0)
                    h_np = agent_hidden.get(agent)
                    h_t = (torch.tensor(h_np, dtype=torch.float32, device=device).unsqueeze(0)
                           if h_np is not None else None)
                    action, log_prob, _, value, _, new_hidden = net.get_action_and_value(
                        obs_t, hidden=h_t)
                    if new_hidden is not None:
                        agent_hidden[agent] = new_hidden.detach().cpu().numpy()[0]
                    if arch == "attention_gru":
                        h_store = h_np if h_np is not None else np.zeros(
                            args.hidden_dim, dtype=np.float32)
                    else:
                        h_store = None
                    buffer.add(agent_id=agent, obs=obs_flat[agent],
                               action=action.item(), log_prob=log_prob.item(),
                               reward=0.0, done=False, value=value.item(),
                               hidden=h_store)

            actions = {}
            buf_len = len(buffer.actions)
            idx = buf_len - len(env.agents)
            agent_indices = {}
            for agent in env.agents:
                actions[agent] = buffer.actions[idx]
                agent_indices[agent] = idx
                idx += 1

            obs_raw, rewards, terms, truncs, infos = env.step(actions)
            obs_flat = flatten_obs(obs_raw) if env.agents else {}

            for agent in rewards:
                ri = infos.get(agent, {}).get("reward_info", {})
                raw_k = ri.get("raw_kill", 0.0)
                role_b = ri.get("role_bonus", 0.0)
                lock_b = ri.get("lock_bonus", 0.0)
                fail_s = ri.get("failure_shared", 0.0)
                total = ri.get("total", rewards[agent])
                done = terms.get(agent, False) or truncs.get(agent, False)
                bi = agent_indices[agent]
                buffer.rewards[bi] = total
                buffer.dones[bi] = done
                buffer.raw_kill_rewards[bi] = raw_k
                buffer.role_bonus_rewards[bi] = role_b
                buffer.lock_bonus_rewards[bi] = lock_b
                buffer.failure_shared_rewards[bi] = fail_s
                ep_return[agent] += total
                ep_raw_kill[agent] += raw_k
                ep_role[agent] += role_b
                ep_lock[agent] += lock_b
                ep_fail[agent] += fail_s

            ep_length += 1
            global_step += len(rewards)

        # GAE
        last_values = {}
        if env.agents:
            with torch.no_grad():
                for agent in env.agents:
                    obs_t = torch.tensor(obs_flat[agent], dtype=torch.float32,
                                         device=device).unsqueeze(0)
                    h_np = agent_hidden.get(agent)
                    h_t = (torch.tensor(h_np, dtype=torch.float32, device=device).unsqueeze(0)
                           if h_np is not None else None)
                    _, _, _, v, _, _ = net.get_action_and_value(obs_t, hidden=h_t)
                    last_values[agent] = v.item()
        adv, returns = buffer.compute_gae(last_values, args.gamma, args.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        tensors = buffer.to_tensors(device)
        all_obs = tensors["obs"]
        all_actions = tensors["actions"]
        all_old_lp = tensors["log_probs"]
        all_hidden = tensors["hidden_states"]
        all_returns = torch.tensor(returns, dtype=torch.float32, device=device)
        all_adv = torch.tensor(adv, dtype=torch.float32, device=device)

        n = len(all_obs)
        p_losses, v_losses, e_list = [], [], []
        for _ in range(args.n_epochs):
            perm = np.random.permutation(n)
            for s in range(0, n, args.batch_size):
                e = min(s + args.batch_size, n)
                mb = torch.tensor(perm[s:e], dtype=torch.long, device=device)
                mb_obs = all_obs[mb]; mb_a = all_actions[mb]
                mb_lp = all_old_lp[mb]; mb_ret = all_returns[mb]; mb_adv = all_adv[mb]
                mb_h = all_hidden[mb].detach() if all_hidden is not None else None
                _, new_lp, entropy, new_val, _, _ = net.get_action_and_value(
                    mb_obs, mb_a, mb_h)
                ratio = torch.exp(new_lp - mb_lp)
                s1 = ratio * mb_adv
                s2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * mb_adv
                pl = -torch.min(s1, s2).mean()
                vl = nn.functional.mse_loss(new_val, mb_ret)
                el = -entropy.mean()
                eff = args.ent_coef
                if args.min_entropy is not None and entropy.mean().item() < args.min_entropy:
                    eff = args.ent_coef * 2.0
                loss = pl + args.vf_coef * vl + eff * el
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                optimizer.step()
                p_losses.append(pl.item()); v_losses.append(vl.item())
                e_list.append(-el.item())

        update_count += 1

        if update_count % args.log_interval == 0:
            elapsed = time.time() - start_time
            sps = global_step / max(elapsed, 1e-6)
            avg_p = float(np.mean(p_losses)); avg_v = float(np.mean(v_losses))
            avg_e = float(np.mean(e_list))
            avg_r = float(np.mean(buffer.rewards))
            avg_rk = float(np.mean(buffer.raw_kill_rewards))
            avg_role = float(np.mean(buffer.role_bonus_rewards))
            avg_lock = float(np.mean(buffer.lock_bonus_rewards))
            avg_fail = float(np.mean(buffer.failure_shared_rewards))

            writer.add_scalar("reward/total", avg_r, global_step)
            writer.add_scalar("reward/raw_kills", avg_rk, global_step)
            writer.add_scalar("reward/role_bonus", avg_role, global_step)
            writer.add_scalar("reward/lock_bonus", avg_lock, global_step)
            writer.add_scalar("reward/failure_shared", avg_fail, global_step)
            writer.add_scalar("reward/aggression", avg_rk, global_step)   # rules #9
            writer.add_scalar("reward/preservation", avg_role + avg_lock, global_step)
            writer.add_scalar("policy/loss", avg_p, global_step)
            writer.add_scalar("policy/entropy", avg_e, global_step)
            writer.add_scalar("value/loss", avg_v, global_step)
            writer.add_scalar("training/sps", sps, global_step)
            if rollout_ep_returns:
                writer.add_scalar("metrics/episode_return",
                                  float(np.mean(rollout_ep_returns)), global_step)
                writer.add_scalar("metrics/episode_length",
                                  float(np.mean(rollout_ep_lengths)), global_step)
            if rollout_ep_stats:
                stats_mean = {k: float(np.mean([s.get(k, 0) or 0 for s in rollout_ep_stats
                                               if isinstance(s.get(k, 0), (int, float))]))
                              for k in ["score", "failures", "kills_archer", "kills_knight",
                                        "attacks_archer", "attacks_knight",
                                        "ammo_pct_expended", "stamina_pct_expended"]}
                for k, v in stats_mean.items():
                    writer.add_scalar(f"metric/{k}", v, global_step)
                writer.add_scalar("metric/ammo_pct_remaining",
                                  1.0 - stats_mean["ammo_pct_expended"], global_step)
                writer.add_scalar("metric/stamina_pct_remaining",
                                  1.0 - stats_mean["stamina_pct_expended"], global_step)

            print(f"[u{update_count:>4d}] step={global_step:>8d} ep={episode_count:>4d} "
                  f"R={avg_r:+.3f} rk={avg_rk:+.3f} role={avg_role:+.4f} "
                  f"lock={avg_lock:+.4f} fail={avg_fail:+.3f} "
                  f"pi={avg_p:.3f} v={avg_v:.3f} ent={avg_e:.3f} SPS={sps:.0f}")

        if update_count % args.save_interval == 0:
            p = os.path.join(ckpt_dir, f"checkpoint_{global_step}.pt")
            torch.save({"model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step, "update_count": update_count,
                        "args": vars(args), "arch": arch}, p)
            print(f"[ckpt] {p}")

    final = os.path.join(ckpt_dir, "final.pt")
    torch.save({"model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step, "update_count": update_count,
                "args": vars(args), "arch": arch}, final)
    print(f"[final] {final}")

    metrics = {
        "game": args.game, "arch": arch, "total_timesteps": global_step,
        "episodes": episode_count,
        "mean_episode_return": float(np.mean(episode_returns[-50:])) if episode_returns else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths[-50:])) if episode_lengths else 0.0,
        "last_episode_stats": last_ep_stats,
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/{args.game}_training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: None)

    writer.close(); env.close()
    print("[train_v3] Done.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
