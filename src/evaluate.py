"""
Evaluate a trained MAPPO checkpoint on KAZ.

Usage:
    ./.venv/bin/python src/evaluate.py --checkpoint models/game1/final.pt --episodes 10
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, ".")
from src.models.mappo_net import MAPPONet
from src.wrappers.kaz_wrapper import KAZWrapper


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MAPPO on KAZ")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to .pt checkpoint file")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", action="store_true", help="Render human-visible window")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def flatten_obs(obs_dict):
    return {a: ob.flatten().astype(np.float32) for a, ob in obs_dict.items()}


def evaluate(args):
    device = torch.device("cuda")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    game_level = train_args["game_level"]
    hidden_dim = train_args.get("hidden_dim", 256)

    print(f"[eval] Checkpoint: {args.checkpoint}")
    print(f"[eval] Game level: {game_level}, trained for {ckpt['global_step']} steps")

    # Environment
    env = KAZWrapper(
        game_level=game_level,
        vector_state=True,
        render_mode="human" if args.render else None,
        seed=args.seed,
    )
    obs_raw, _ = env.reset()
    obs_flat = flatten_obs(obs_raw)
    obs_dim = list(obs_flat.values())[0].shape[0]
    act_dim = env.action_space(env.agents[0]).n

    # Network
    net = MAPPONet(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # Run evaluation episodes
    all_returns = []
    all_lengths = []
    all_aggression = []
    all_preservation = []

    for ep in range(args.episodes):
        obs_raw, _ = env.reset()
        obs_flat = flatten_obs(obs_raw)
        ep_return = defaultdict(float)
        ep_aggr = defaultdict(float)
        ep_pres = defaultdict(float)
        ep_len = 0

        while env.agents:
            actions = {}
            with torch.no_grad():
                for agent in env.agents:
                    obs_t = torch.tensor(
                        obs_flat[agent], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    action, _, _, _, _ = net.get_action_and_value(obs_t)
                    actions[agent] = action.item()

            obs_raw, rewards, terms, truncs, infos = env.step(actions)
            obs_flat = flatten_obs(obs_raw) if env.agents else {}

            for agent in rewards:
                ri = infos.get(agent, {}).get("reward_info", {})
                ep_return[agent] += ri.get("total", rewards[agent])
                ep_aggr[agent] += ri.get("aggression", 0.0)
                ep_pres[agent] += ri.get("preservation", 0.0)
            ep_len += 1

        mean_ret = float(np.mean(list(ep_return.values())))
        mean_aggr = float(np.mean(list(ep_aggr.values())))
        mean_pres = float(np.mean(list(ep_pres.values())))
        all_returns.append(mean_ret)
        all_lengths.append(ep_len)
        all_aggression.append(mean_aggr)
        all_preservation.append(mean_pres)

        print(
            f"  Episode {ep+1:>3d}: return={mean_ret:+.2f}  "
            f"aggr={mean_aggr:+.2f}  pres={mean_pres:+.2f}  len={ep_len}"
        )

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print(f"Evaluation Summary ({args.episodes} episodes, Game {game_level})")
    print("=" * 60)
    print(f"  Mean return:      {np.mean(all_returns):+.3f} +/- {np.std(all_returns):.3f}")
    print(f"  Mean aggression:  {np.mean(all_aggression):+.3f}")
    print(f"  Mean preservation:{np.mean(all_preservation):+.3f}")
    print(f"  Mean ep length:   {np.mean(all_lengths):.1f}")
    print(f"  Kill density:     {np.mean(all_aggression) / max(np.mean(all_lengths), 1):.5f}")

    # Save eval results
    results = {
        "checkpoint": args.checkpoint,
        "game_level": game_level,
        "episodes": args.episodes,
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "mean_aggression": float(np.mean(all_aggression)),
        "mean_preservation": float(np.mean(all_preservation)),
        "mean_episode_length": float(np.mean(all_lengths)),
        "kill_density": float(np.mean(all_aggression) / max(np.mean(all_lengths), 1)),
    }
    out_path = f"results/game{game_level}_eval_results.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
