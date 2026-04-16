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

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from src.models.mappo_net import MAPPONet
from src.utils import get_device, device_info
from src.wrappers.kaz_wrapper import KAZWrapper


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MAPPO on KAZ")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to .pt checkpoint file")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", action="store_true", help="Render human-visible window")
    p.add_argument("--record", action="store_true",
                    help="Record episodes as mp4 to results/game<N>_demo/")
    p.add_argument("--deterministic", action="store_true",
                    help="Use greedy (argmax) policy instead of sampling")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def flatten_obs(obs_dict):
    return {a: ob.flatten().astype(np.float32) for a, ob in obs_dict.items()}


def evaluate(args):
    device = get_device()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    game_level = train_args["game_level"]
    hidden_dim = train_args.get("hidden_dim", 256)
    arch = train_args.get("arch", "mlp")
    num_entities = train_args.get("num_entities", 27)
    entity_dim = train_args.get("entity_dim", 5)
    num_heads = train_args.get("num_heads", 4)

    death_penalty = train_args.get("death_penalty", 0.0)

    print(f"[eval] Device: {device_info(device)}")
    print(f"[eval] Death penalty: {death_penalty}")
    print(f"[eval] Checkpoint: {args.checkpoint}")
    print(f"[eval] Game level: {game_level}, arch: {arch}, trained for {ckpt['global_step']} steps")

    # Environment
    render_mode = None
    if args.render:
        render_mode = "human"
    elif args.record:
        render_mode = "rgb_array"

    env = KAZWrapper(
        game_level=game_level,
        vector_state=True,
        render_mode=render_mode,
        seed=args.seed,
        death_penalty=death_penalty,
    )

    # Recording setup
    video_dir = None
    if args.record:
        video_dir = f"results/game{game_level}_demo"
        os.makedirs(video_dir, exist_ok=True)
        print(f"[eval] Recording to {video_dir}/")
    obs_raw, _ = env.reset()
    obs_flat = flatten_obs(obs_raw)
    obs_dim = list(obs_flat.values())[0].shape[0]
    act_dim = env.action_space(env.agents[0]).n

    # Network
    net = MAPPONet(
        obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim,
        arch=arch, num_entities=num_entities,
        entity_dim=entity_dim, num_heads=num_heads,
    ).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # Run evaluation episodes
    all_returns = []
    all_lengths = []
    all_raw_kills = []
    all_aggression = []
    all_preservation = []
    all_death = []

    for ep in range(args.episodes):
        obs_raw, _ = env.reset()
        obs_flat = flatten_obs(obs_raw)
        ep_return = defaultdict(float)
        ep_aggr = defaultdict(float)
        ep_raw_kills = defaultdict(float)
        ep_pres = defaultdict(float)
        ep_death = defaultdict(float)
        ep_len = 0
        frames = []
        # Per-agent GRU hidden states; reset each episode
        ep_hidden = {}

        if args.record:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        while env.agents:
            actions = {}
            with torch.no_grad():
                for agent in env.agents:
                    obs_t = torch.tensor(
                        obs_flat[agent], dtype=torch.float32, device=device
                    ).unsqueeze(0)

                    # Retrieve (or zero-init) GRU hidden state
                    h_np = ep_hidden.get(agent)
                    h_t = (
                        torch.tensor(h_np, dtype=torch.float32, device=device).unsqueeze(0)
                        if h_np is not None else None
                    )

                    if args.deterministic:
                        out = net.forward(obs_t, hidden=h_t)
                        action = out["logits"].argmax(dim=-1)
                        new_hidden = out["hidden"]
                    else:
                        action, _, _, _, _, new_hidden = net.get_action_and_value(
                            obs_t, hidden=h_t
                        )

                    # Persist updated GRU state (detached)
                    if new_hidden is not None:
                        ep_hidden[agent] = new_hidden.detach().cpu().numpy()[0]

                    actions[agent] = action.item()

            obs_raw, rewards, terms, truncs, infos = env.step(actions)
            obs_flat = flatten_obs(obs_raw) if env.agents else {}

            if args.record:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            for agent in rewards:
                ri = infos.get(agent, {}).get("reward_info", {})
                ep_return[agent] += ri.get("total", rewards[agent])
                ep_raw_kills[agent] += ri.get("raw_kill", 0.0)
                ep_aggr[agent] += ri.get("aggression", 0.0)
                ep_pres[agent] += ri.get("preservation", 0.0)
                ep_death[agent] += ri.get("death_penalty", 0.0)
            ep_len += 1

        mean_ret = float(np.mean(list(ep_return.values())))
        mean_raw_k = float(np.mean(list(ep_raw_kills.values())))
        mean_aggr = float(np.mean(list(ep_aggr.values())))
        mean_pres = float(np.mean(list(ep_pres.values())))
        mean_death = float(np.mean(list(ep_death.values()))) if ep_death else 0.0
        all_returns.append(mean_ret)
        all_lengths.append(ep_len)
        all_raw_kills.append(mean_raw_k)
        all_aggression.append(mean_aggr)
        all_preservation.append(mean_pres)
        all_death.append(mean_death)

        # Save video if recording
        if args.record and frames:
            video_path = os.path.join(video_dir, f"episode_{ep+1}.mp4")
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_vid = cv2.VideoWriter(video_path, fourcc, 30, (w, h))
            for frame in frames:
                writer_vid.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer_vid.release()
            print(f"  Episode {ep+1:>3d}: return={mean_ret:+.2f}  "
                  f"aggr={mean_aggr:+.2f}  pres={mean_pres:+.2f}  "
                  f"len={ep_len}  -> {video_path}")
        else:
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
    print(f"  Mean raw kills:   {np.mean(all_raw_kills):+.3f}")
    print(f"  Mean aggression:  {np.mean(all_aggression):+.3f}")
    print(f"  Mean preservation:{np.mean(all_preservation):+.3f}")
    print(f"  Mean ep length:   {np.mean(all_lengths):.1f}")
    print(f"  Raw kill density: {np.mean(all_raw_kills) / max(np.mean(all_lengths), 1):.5f}")
    print(f"  Kill density:     {np.mean(all_aggression) / max(np.mean(all_lengths), 1):.5f}")

    # Save eval results
    results = {
        "checkpoint": args.checkpoint,
        "game_level": game_level,
        "episodes": args.episodes,
        "deterministic": args.deterministic,
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "mean_raw_kills": float(np.mean(all_raw_kills)),
        "mean_aggression": float(np.mean(all_aggression)),
        "mean_preservation": float(np.mean(all_preservation)),
        "mean_episode_length": float(np.mean(all_lengths)),
        "raw_kill_density": float(np.mean(all_raw_kills) / max(np.mean(all_lengths), 1)),
        "kill_density": float(np.mean(all_aggression) / max(np.mean(all_lengths), 1)),
        "mean_death_penalty": float(np.mean(all_death)),
    }
    det_suffix = "_det" if args.deterministic else ""
    out_path = f"results/game{game_level}_eval_results{det_suffix}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
