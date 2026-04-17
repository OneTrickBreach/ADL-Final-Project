"""V3 evaluation harness.

Supports all 7 game levels. Heuristic for G0/G1a/G1b, learned for G2-G5.
Uses fixed per-episode seeds (args.seed + ep_idx) so zombie patterns are
identical across games for a given episode index (§4.5 / plan §4.6).

Usage:
    ./.venv/bin/python src/evaluate_v3.py --game g0 --episodes 10 --seed 42
    ./.venv/bin/python src/evaluate_v3.py --game g2 --checkpoint models/v3/g2/final.pt \
        --episodes 10 --seed 42 [--deterministic] [--record]
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
from src.utils import get_device, device_info
from src.wrappers.kaz_wrapper_v3 import KAZWrapperV3, EXTRA_DIM
from src.policies.heuristic import heuristic_actions_all


HEURISTIC_GAMES = {"g0", "g1a", "g1b"}
LEARNED_GAMES = {"g2", "g3", "g4", "g5"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--game", type=str, required=True,
                   choices=sorted(HEURISTIC_GAMES | LEARNED_GAMES))
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--ammo_mode", type=str, default=None,
                   choices=["global", "individual"],
                   help="Required for G2-G5 heuristic eval; for learned games, "
                        "read from checkpoint args if absent.")
    p.add_argument("--duration_seconds", type=int, default=30)
    p.add_argument("--output_suffix", type=str, default="")
    return p.parse_args()


def flatten_obs(obs_dict):
    return {a: ob.flatten().astype(np.float32) for a, ob in obs_dict.items()}


def _resolve_ammo_mode(args):
    if args.game in ("g0", "g1a", "g1b"):
        return None
    # Learned game: prefer checkpoint's ammo_mode, else CLI, else read ammo_mode.txt
    if args.checkpoint and os.path.isfile(args.checkpoint):
        try:
            ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            am = ck.get("args", {}).get("ammo_mode")
            if am:
                return am
        except Exception:
            pass
    if args.ammo_mode:
        return args.ammo_mode
    pth = "results/v3/ammo_mode.txt"
    if os.path.isfile(pth):
        with open(pth) as f:
            return f.read().strip()
    return "individual"


def _build_env(args, ammo_mode, render_mode=None, seed=None):
    return KAZWrapperV3(
        game_level=args.game,
        duration_seconds=args.duration_seconds,
        ammo_mode_override=ammo_mode,
        vector_state=True,
        render_mode=render_mode,
        seed=seed,
    )


def _load_net(args, env, device):
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ta = ck["args"]
    arch = ck.get("arch", ta.get("arch", "attention"))
    hidden_dim = ta.get("hidden_dim", 256)
    ne = ta.get("num_entities", 27); ed = ta.get("entity_dim", 5)
    nh = ta.get("num_heads", 4)

    obs_raw, _ = env.reset(seed=args.seed)
    obs_flat = flatten_obs(obs_raw)
    obs_dim = list(obs_flat.values())[0].shape[0]
    act_dim = env.action_space(env.agents[0]).n
    net = MAPPONet(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim,
                   arch=arch, num_entities=ne, entity_dim=ed, num_heads=nh,
                   extra_dim=EXTRA_DIM).to(device)
    net.load_state_dict(ck["model_state_dict"])
    net.eval()
    return net, arch


def evaluate(args):
    device = get_device()
    print(f"[eval_v3] device={device_info(device)} game={args.game} "
          f"episodes={args.episodes} det={args.deterministic}")

    ammo_mode = _resolve_ammo_mode(args)
    use_heuristic = args.game in HEURISTIC_GAMES

    render_mode = "rgb_array" if args.record else None
    env = _build_env(args, ammo_mode, render_mode=render_mode, seed=args.seed)

    net, arch = (None, None)
    if not use_heuristic:
        if not args.checkpoint:
            raise ValueError(f"--checkpoint required for game {args.game}")
        net, arch = _load_net(args, env, device)

    video_dir = None
    if args.record:
        video_dir = f"results/v3/{args.game}_demo"
        os.makedirs(video_dir, exist_ok=True)

    per_episode = []

    for ep in range(args.episodes):
        seed = args.seed + ep
        obs_raw, _ = env.reset(seed=seed)
        obs_flat = flatten_obs(obs_raw)
        ep_return = defaultdict(float)
        ep_raw = defaultdict(float)
        ep_role = defaultdict(float)
        ep_lock = defaultdict(float)
        ep_fail = defaultdict(float)
        ep_hidden = {}
        frames = []

        if args.record:
            fr = env.render()
            if fr is not None:
                frames.append(fr)

        while env.agents:
            if use_heuristic:
                actions = heuristic_actions_all(env)
            else:
                actions = {}
                with torch.no_grad():
                    for agent in env.agents:
                        obs_t = torch.tensor(obs_flat[agent], dtype=torch.float32,
                                             device=device).unsqueeze(0)
                        h_np = ep_hidden.get(agent)
                        h_t = (torch.tensor(h_np, dtype=torch.float32,
                                            device=device).unsqueeze(0)
                               if h_np is not None else None)
                        if args.deterministic:
                            out = net.forward(obs_t, hidden=h_t)
                            act = out["logits"].argmax(dim=-1)
                            nh = out["hidden"]
                        else:
                            act, _, _, _, _, nh = net.get_action_and_value(
                                obs_t, hidden=h_t)
                        if nh is not None:
                            ep_hidden[agent] = nh.detach().cpu().numpy()[0]
                        actions[agent] = act.item()

            obs_raw, rewards, terms, truncs, infos = env.step(actions)
            obs_flat = flatten_obs(obs_raw) if env.agents else {}
            if args.record:
                fr = env.render()
                if fr is not None:
                    frames.append(fr)
            for a, r in rewards.items():
                ri = infos.get(a, {}).get("reward_info", {})
                ep_return[a] += ri.get("total", r)
                ep_raw[a] += ri.get("raw_kill", 0.0)
                ep_role[a] += ri.get("role_bonus", 0.0)
                ep_lock[a] += ri.get("lock_bonus", 0.0)
                ep_fail[a] += ri.get("failure_shared", 0.0)

        stats = env.get_episode_stats()
        stats["mean_return"] = float(np.mean(list(ep_return.values()))) if ep_return else 0.0
        stats["seed"] = seed
        per_episode.append(stats)

        if args.record and frames:
            try:
                import cv2
                path = os.path.join(video_dir, f"episode_{ep+1}_seed{seed}.mp4")
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vid = cv2.VideoWriter(path, fourcc, 15, (w, h))
                for f in frames:
                    vid.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                vid.release()
                print(f"  ep {ep+1:>3d} seed={seed} score={stats['score']} "
                      f"failures={stats['failures']} -> {path}")
            except Exception as e:
                print(f"  [warn] video write failed: {e}")
        else:
            print(f"  ep {ep+1:>3d} seed={seed} score={stats['score']:>3d} "
                  f"kills={stats['kills_total']:>2d} (A={stats['kills_archer']} "
                  f"K={stats['kills_knight']}) failures={stats['failures']:>2d} "
                  f"attacks(A/K)={stats['attacks_archer']}/{stats['attacks_knight']} "
                  f"ammo_exp={stats['ammo_pct_expended']:.2f} "
                  f"stam_exp={stats['stamina_pct_expended']:.2f}")

    env.close()

    scores = [s["score"] for s in per_episode]
    ammo_per = [s.get("ammo_pct_expended_per_archer", []) for s in per_episode]
    # flatten per-archer across episodes where individual mode; gives mean per archer
    archer_means = []
    if ammo_mode == "individual" and ammo_per and ammo_per[0]:
        n_arch = len(ammo_per[0])
        archer_means = []
        for i in range(n_arch):
            vals = [ep[i] for ep in ammo_per if len(ep) > i and ep[i] is not None]
            archer_means.append(float(np.mean(vals)) if vals else None)

    stam_per = [s.get("stamina_pct_expended_per_knight", []) for s in per_episode]
    knight_means = []
    if stam_per and stam_per[0]:
        n_k = len(stam_per[0])
        for i in range(n_k):
            vals = [ep[i] for ep in stam_per if len(ep) > i]
            knight_means.append(float(np.mean(vals)) if vals else 0.0)

    summary = {
        "game": args.game,
        "episodes": args.episodes,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "ammo_mode": ammo_mode,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "stamina_pct_expended_mean": float(np.mean(
            [s["stamina_pct_expended"] for s in per_episode])),
        "stamina_pct_expended_per_knight": knight_means,
        "ammo_pct_expended_mean": float(np.mean(
            [s["ammo_pct_expended"] for s in per_episode])),
        "ammo_pct_expended_per_archer": archer_means,
        "kills_archer_mean": float(np.mean([s["kills_archer"] for s in per_episode])),
        "kills_knight_mean": float(np.mean([s["kills_knight"] for s in per_episode])),
        "attacks_archer_mean": float(np.mean([s["attacks_archer"] for s in per_episode])),
        "attacks_knight_mean": float(np.mean([s["attacks_knight"] for s in per_episode])),
        "failures_mean": float(np.mean([s["failures"] for s in per_episode])),
        "episode_length_mean": float(np.mean([s["episode_length"] for s in per_episode])),
        "per_episode": per_episode,
    }

    det = "_det" if args.deterministic else ""
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    os.makedirs("results/v3", exist_ok=True)
    path = f"results/v3/{args.game}_eval_results{det}{suffix}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: None)
    print(f"\n[eval_v3] score={summary['score_mean']:.2f} ± {summary['score_std']:.2f} "
          f"-> {path}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
