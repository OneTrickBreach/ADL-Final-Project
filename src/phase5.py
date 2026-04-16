"""
Phase 5: Evaluation & Explainability

Generates all Phase 5 artifacts:
  1. Kill density evolution bar chart  → results/kill_density_evolution.png
  2. Collapse / stress-test graph      → results/collapse_graph.png
  3. Saliency comparison               → results/saliency_comparison.png
  4. Side-by-side demo video           → results/demo_sidebyside.mp4

Usage:
    ./.venv/bin/python src/phase5.py
"""

import json
import os
import sys
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

sys.path.insert(0, ".")
from src.models.mappo_net import MAPPONet
from src.utils import get_device, device_info
from src.wrappers.kaz_wrapper import KAZWrapper

os.makedirs("results", exist_ok=True)
DEVICE = get_device()
print(f"[phase5] Device: {device_info(DEVICE)}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_net(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    ta = ckpt["args"]
    # Derive obs_dim from the training environment so it's always correct
    death_penalty = ta.get("death_penalty", 0.0)
    _env = KAZWrapper(game_level=ta["game_level"], vector_state=True,
                      death_penalty=death_penalty)
    _obs, _ = _env.reset()
    obs_dim = list(_obs.values())[0].flatten().shape[0]
    act_dim = _env.action_space(_env.agents[0]).n
    _env.close()
    net = MAPPONet(
        obs_dim=obs_dim, act_dim=act_dim,
        hidden_dim=ta.get("hidden_dim", 256),
        arch=ta.get("arch", "mlp"),
        num_entities=ta.get("num_entities", 27),
        entity_dim=ta.get("entity_dim", 5),
        num_heads=ta.get("num_heads", 4),
    ).to(DEVICE)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    ta["_obs_dim"] = obs_dim  # cache for later use
    ta["_death_penalty"] = death_penalty  # cache for downstream callers
    return net, ta


def flatten(obs_dict):
    return {a: ob.flatten().astype(np.float32) for a, ob in obs_dict.items()}


def run_episodes(net, game_level, episodes=5, spawn_rate=20,
                 seed=42, deterministic=True, death_penalty=0.0):
    """Return list of (raw_kills, ep_len) per episode."""
    env = KAZWrapper(game_level=game_level, vector_state=True,
                     spawn_rate=spawn_rate, seed=seed,
                     death_penalty=death_penalty)
    results = []
    ep_hidden = {}
    for _ in range(episodes):
        obs_raw, _ = env.reset()
        obs_flat = flatten(obs_raw)
        ep_hidden = {}
        kills, length = 0.0, 0
        while env.agents:
            actions = {}
            with torch.no_grad():
                for agent in env.agents:
                    obs_t = torch.tensor(
                        obs_flat[agent], dtype=torch.float32, device=DEVICE
                    ).unsqueeze(0)
                    h_np = ep_hidden.get(agent)
                    h_t = (torch.tensor(h_np, dtype=torch.float32, device=DEVICE
                                        ).unsqueeze(0) if h_np is not None else None)
                    if deterministic:
                        out = net.forward(obs_t, hidden=h_t)
                        action = int(out["logits"].argmax(dim=-1).item())
                        new_h = out["hidden"]
                    else:
                        action, _, _, _, _, new_h = net.get_action_and_value(
                            obs_t, hidden=h_t)
                        action = int(action.item())
                    if new_h is not None:
                        ep_hidden[agent] = new_h.detach().cpu().numpy()[0]
                    actions[agent] = action
            obs_raw, rewards, terms, truncs, infos = env.step(actions)
            obs_flat = flatten(obs_raw) if env.agents else {}
            for ag in rewards:
                kills += infos.get(ag, {}).get("reward_info", {}).get("raw_kill", 0.0)
            length += 1
        results.append((kills, length))
    env.close()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Kill Density Evolution (uses existing eval JSONs)
# ─────────────────────────────────────────────────────────────────────────────

def part1_kill_density_evolution():
    print("[phase5] Part 1: Kill density evolution chart...")
    games = [1, 2, 3, 4, 5]
    labels = ["G1\nGreedy\nSoldier", "G2\nRisk\nAvoider",
              "G3\nFully\nPassive", "G4\nRecovering\nCooperator",
              "G5\nFire\nDiscipline"]
    colors = ["#e74c3c", "#e67e22", "#95a5a6", "#3498db", "#2ecc71"]

    raw_kill_densities, kill_densities, preservations = [], [], []
    for g in games:
        path = f"results/game{g}_eval_results.json"
        with open(path) as f:
            d = json.load(f)
        raw_kill_densities.append(d["raw_kill_density"])
        kill_densities.append(d["kill_density"])
        preservations.append(d["mean_preservation"])

    x = np.arange(len(games))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("The Evolution of Fire Discipline — G1→G5 Ablation",
                 fontsize=14, fontweight="bold")

    # Left: raw kill density
    ax = axes[0]
    bars = ax.bar(x, raw_kill_densities, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Raw Kill Density (kills / step)", fontsize=11)
    ax.set_title("Raw Kill Density Across Games", fontsize=12)
    for bar, val in zip(bars, raw_kill_densities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                f"{val:.5f}", ha="center", va="bottom", fontsize=8)
    # Annotate the G1→G3 collapse and G4→G5 recovery
    ax.annotate("Behavioral\nCollapse", xy=(1, raw_kill_densities[1]),
                xytext=(1.5, 0.007), fontsize=8, color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c"))
    ax.annotate("Recovery", xy=(3, raw_kill_densities[3]),
                xytext=(3.3, 0.005), fontsize=8, color="#2ecc71",
                arrowprops=dict(arrowstyle="->", color="#2ecc71"))

    # Right: preservation (team reward)
    ax2 = axes[1]
    bars2 = ax2.bar(x, preservations, color=colors, edgecolor="black",
                    linewidth=0.8, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Mean Preservation Reward", fontsize=11)
    ax2.set_title("Preservation (Team Health) Signal", fontsize=12)
    for bar, val in zip(bars2, preservations):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.annotate("60/40 team\nreward\nactivated →", xy=(3, preservations[3]),
                 xytext=(1.2, preservations[3] * 0.7), fontsize=8, color="#3498db",
                 arrowprops=dict(arrowstyle="->", color="#3498db"))

    plt.tight_layout()
    out = "results/kill_density_evolution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Collapse / Stress-Test Graph
# ─────────────────────────────────────────────────────────────────────────────

def part2_collapse_graph():
    print("[phase5] Part 2: Stress-test / collapse graph...")
    net_g1, ta_g1 = load_net("models/game1/final.pt")
    net_g5, ta_g5 = load_net("models/game5/final.pt")
    dp_g1 = ta_g1.get("_death_penalty", 0.0)
    dp_g5 = ta_g5.get("_death_penalty", 0.0)

    # Stress via spawn_rate: lower = faster zombie spawning = harder
    # 20=normal(training), 14=1.5x pressure, 10=2x pressure, 7=3x pressure
    spawn_rates = [28, 20, 14, 10]
    x_labels = ["0.7x", "1.0x\n(training)", "1.4x", "2.0x"]
    episodes_per_rate = 5

    g1_kills, g1_lens = [], []
    g5_kills, g5_lens = [], []

    for sr in spawn_rates:
        print(f"  spawn_rate={sr}...")
        r1 = run_episodes(net_g1, game_level=1, episodes=episodes_per_rate,
                          spawn_rate=sr, seed=77, death_penalty=dp_g1)
        r5 = run_episodes(net_g5, game_level=5, episodes=episodes_per_rate,
                          spawn_rate=sr, seed=77, death_penalty=dp_g5)
        g1_kills.append(np.mean([k for k, _ in r1]))
        g1_lens.append(np.mean([l for _, l in r1]))
        g5_kills.append(np.mean([k for k, _ in r5]))
        g5_lens.append(np.mean([l for _, l in r5]))

    # Normalise: kills-per-step at each density
    g1_density = [k / max(l, 1) for k, l in zip(g1_kills, g1_lens)]
    g5_density = [k / max(l, 1) for k, l in zip(g5_kills, g5_lens)]

    # Normalise to performance at training spawn_rate=20
    baseline_idx = spawn_rates.index(20)
    g1_norm = [v / max(g1_density[baseline_idx], 1e-9) for v in g1_density]
    g5_norm = [v / max(g5_density[baseline_idx], 1e-9) for v in g5_density]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Stress Test: Performance Under Increasing Zombie Spawn Pressure\n"
        "(G1 on its training env: no constraints  |  G5 on its training env: fog + ammo + stamina)",
        fontsize=11, fontweight="bold")

    x = np.arange(len(spawn_rates))

    ax = axes[0]
    ax.plot(x, g1_kills, "o-", color="#e74c3c", lw=2,
            label="G1 — Greedy Soldier")
    ax.plot(x, g5_kills, "s-", color="#2ecc71", lw=2,
            label="G5 — Fire Discipline")
    ax.axvline(baseline_idx, color="gray", linestyle="--", lw=1,
               label="Training pressure")
    ax.set_xlabel("Zombie Spawn Pressure (relative to training)", fontsize=11)
    ax.set_ylabel("Mean Raw Kills / Episode", fontsize=11)
    ax.set_title("Raw Kills vs Spawn Pressure\n(G1: unconstrained env  |  G5: fog+ammo+stamina env)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)

    ax2 = axes[1]
    ax2.plot(x, g1_norm, "o-", color="#e74c3c", lw=2,
             label="G1 — Greedy Soldier")
    ax2.plot(x, g5_norm, "s-", color="#2ecc71", lw=2,
             label="G5 — Fire Discipline")
    ax2.axvline(baseline_idx, color="gray", linestyle="--", lw=1,
                label="Training pressure")
    ax2.axhline(1.0, color="black", linestyle=":", lw=0.8)
    ax2.set_xlabel("Zombie Spawn Pressure (relative to training)", fontsize=11)
    ax2.set_ylabel("Kill Density (normalised to 1.0x training)", fontsize=11)
    ax2.set_title("Normalised Kill Density — G5 degrades less\ndespite harder base environment",
                  fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=9)

    plt.tight_layout()
    out = "results/collapse_graph.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Saliency / Attention Maps
# ─────────────────────────────────────────────────────────────────────────────

def part3_saliency():
    print("[phase5] Part 3: Saliency maps...")
    net_g1, ta_g1 = load_net("models/game1/final.pt")
    net_g5, ta_g5 = load_net("models/game5/final.pt")

    ENTITY_DIM = 5
    g1_entities = ta_g1["_obs_dim"] // ENTITY_DIM
    g5_entities = ta_g5["_obs_dim"] // ENTITY_DIM
    NUM_ENTITIES = max(g1_entities, g5_entities)  # for shared plot width
    N_STEPS = 200  # steps to average over

    # ── G1: gradient-based saliency ──────────────────────────────────────
    g1_saliency = np.zeros(g1_entities)
    env = KAZWrapper(game_level=1, vector_state=True, seed=11,
                      death_penalty=ta_g1.get("_death_penalty", 0.0))
    obs_raw, _ = env.reset()
    obs_flat = flatten(obs_raw)
    step_count = 0

    while step_count < N_STEPS and env.agents:
        for agent in list(env.agents)[:1]:  # single agent for saliency
            obs_np = obs_flat[agent]
            obs_t = torch.tensor(obs_np, dtype=torch.float32,
                                 device=DEVICE).unsqueeze(0).requires_grad_(True)
            with torch.enable_grad():
                out = net_g1.forward(obs_t)
                logits = out["logits"]
                # Saliency w.r.t. the chosen action
                chosen = logits.argmax(dim=-1)
                logits[0, chosen].backward()
            grad = obs_t.grad.detach().abs().cpu().numpy()[0]
            per_entity = grad.reshape(g1_entities, ENTITY_DIM).sum(axis=-1)
            # Weight by entity presence (non-zero = active)
            presence = (obs_np.reshape(g1_entities, ENTITY_DIM).sum(axis=-1) != 0
                        ).astype(float)
            g1_saliency += per_entity * presence
        # Step with deterministic actions
        with torch.no_grad():
            actions = {}
            for ag in env.agents:
                obs_t2 = torch.tensor(obs_flat[ag], dtype=torch.float32,
                                      device=DEVICE).unsqueeze(0)
                actions[ag] = int(net_g1.forward(obs_t2)["logits"].argmax(-1).item())
        obs_raw, _, _, _, _ = env.step(actions)
        obs_flat = flatten(obs_raw) if env.agents else {}
        step_count += 1
        if not env.agents:
            obs_raw, _ = env.reset()
            obs_flat = flatten(obs_raw)
    env.close()
    g1_saliency /= max(step_count, 1)

    # ── G5: attention weight visualization ──────────────────────────────
    g5_attention = np.zeros((g5_entities, g5_entities))
    attn_cache = {}

    def attn_hook(module, input, output):
        if output[1] is not None:
            attn_cache["w"] = output[1].detach().cpu().numpy()

    handle = net_g5.entity_attention.attention.register_forward_hook(attn_hook)

    env5 = KAZWrapper(game_level=5, vector_state=True, seed=11,
                       death_penalty=ta_g5.get("_death_penalty", 0.0))
    obs_raw, _ = env5.reset()
    obs_flat5 = flatten(obs_raw)
    ep_hidden5 = {}
    step_count5 = 0

    while step_count5 < N_STEPS and env5.agents:
        for agent in list(env5.agents)[:1]:
            obs_t = torch.tensor(obs_flat5[agent], dtype=torch.float32,
                                 device=DEVICE).unsqueeze(0)
            h_np = ep_hidden5.get(agent)
            h_t = (torch.tensor(h_np, dtype=torch.float32,
                                device=DEVICE).unsqueeze(0) if h_np is not None else None)
            with torch.no_grad():
                out = net_g5.forward(obs_t, hidden=h_t)
                if out["hidden"] is not None:
                    ep_hidden5[agent] = out["hidden"].detach().cpu().numpy()[0]
            if "w" in attn_cache:
                # attn_cache["w"]: (batch, num_heads, tgt, src) or (batch, tgt, src)
                w = attn_cache.pop("w")
                if w.ndim == 4:
                    w = w.mean(axis=1)  # avg over heads → (batch, 27, 27)
                g5_attention += w[0]  # (27, 27)
        with torch.no_grad():
            actions5 = {}
            for ag in env5.agents:
                obs_t2 = torch.tensor(obs_flat5[ag], dtype=torch.float32,
                                      device=DEVICE).unsqueeze(0)
                h_np2 = ep_hidden5.get(ag)
                h_t2 = (torch.tensor(h_np2, dtype=torch.float32,
                                     device=DEVICE).unsqueeze(0)
                        if h_np2 is not None else None)
                out2 = net_g5.forward(obs_t2, hidden=h_t2)
                actions5[ag] = int(out2["logits"].argmax(-1).item())
                if out2["hidden"] is not None:
                    ep_hidden5[ag] = out2["hidden"].detach().cpu().numpy()[0]
        obs_raw, _, _, _, _ = env5.step(actions5)
        obs_flat5 = flatten(obs_raw) if env5.agents else {}
        step_count5 += 1
        if not env5.agents:
            obs_raw, _ = env5.reset()
            obs_flat5 = flatten(obs_raw)
            ep_hidden5 = {}
    handle.remove()
    env5.close()
    g5_attention /= max(step_count5, 1)
    if g5_attention.sum() < 1e-6:
        print("  WARNING: G5 attention matrix is all-zeros — PyTorch fast path may have "
              "skipped weight computation. Visualization will be uninformative.")
    else:
        print(f"  G5 attention: captured over {step_count5} steps, "
              f"sum={g5_attention.sum():.3f}")

    # Column-wise sum → how much each entity slot is "attended to"
    g5_attended = g5_attention.sum(axis=0)  # (g5_entities,)

    # ── Visualise ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Explainability: Where Do Agents Look?",
                 fontsize=14, fontweight="bold")

    # G1 saliency bar chart
    ax1 = axes[0]
    norm_g1 = g1_saliency / (g1_saliency.max() + 1e-9)
    colors_g1 = ["#e74c3c" if v > 0.5 else "#f39c12" if v > 0.2 else "#bdc3c7"
                 for v in norm_g1]
    ax1.bar(range(g1_entities), norm_g1, color=colors_g1, edgecolor="none")
    ax1.set_xticks(range(0, g1_entities, 4))
    ax1.set_xticklabels([f"E{i}" for i in range(0, g1_entities, 4)], fontsize=8)
    ax1.set_xlabel(f"Entity Slot (n={g1_entities})", fontsize=10)
    ax1.set_ylabel("Normalised Gradient Saliency", fontsize=10)
    ax1.set_title("G1 (MLP) — Gradient Saliency\n\"Nearest-enemy tunnel vision\"",
                  fontsize=10)
    high_patch = mpatches.Patch(color="#e74c3c", label="High saliency (>0.5)")
    mid_patch = mpatches.Patch(color="#f39c12", label="Medium (>0.2)")
    low_patch = mpatches.Patch(color="#bdc3c7", label="Low")
    ax1.legend(handles=[high_patch, mid_patch, low_patch], fontsize=7)

    # G5 attention column sum bar chart
    ax2 = axes[1]
    norm_g5 = g5_attended / (g5_attended.max() + 1e-9)
    colors_g5 = ["#2ecc71" if v > 0.5 else "#3498db" if v > 0.2 else "#bdc3c7"
                 for v in norm_g5]
    ax2.bar(range(g5_entities), norm_g5, color=colors_g5, edgecolor="none")
    ax2.set_xticks(range(0, g5_entities, 4))
    ax2.set_xticklabels([f"E{i}" for i in range(0, g5_entities, 4)], fontsize=8)
    ax2.set_xlabel(f"Entity Slot (n={g5_entities})", fontsize=10)
    ax2.set_ylabel("Normalised Attention Weight (column sum)", fontsize=10)
    ax2.set_title("G5 (Attn+GRU) — Self-Attention\n\"Distributed: teammates + threats\"",
                  fontsize=10)
    high_patch2 = mpatches.Patch(color="#2ecc71", label="High attention (>0.5)")
    mid_patch2 = mpatches.Patch(color="#3498db", label="Medium (>0.2)")
    low_patch2 = mpatches.Patch(color="#bdc3c7", label="Low")
    ax2.legend(handles=[high_patch2, mid_patch2, low_patch2], fontsize=7)

    # G5 attention heatmap
    ax3 = axes[2]
    im = ax3.imshow(g5_attention, cmap="hot", aspect="auto")
    ax3.set_xlabel("Source Entity (attended to)", fontsize=10)
    ax3.set_ylabel("Query Entity (attending)", fontsize=10)
    ax3.set_title(f"G5 Attention Matrix (avg over {N_STEPS} steps)", fontsize=10)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_xticks(range(0, g5_entities, 6))
    ax3.set_yticks(range(0, g5_entities, 6))

    plt.tight_layout()
    out = "results/saliency_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: Side-by-Side Demo Video
# ─────────────────────────────────────────────────────────────────────────────

def part4_sidebyside():
    print("[phase5] Part 4: Side-by-side demo video...")

    net_g1, ta_g1 = load_net("models/game1/final.pt")
    net_g5, ta_g5 = load_net("models/game5/final.pt")

    SEED = 55
    MAX_FRAMES = 900
    FPS = 30
    EPISODES = 3
    LABEL_H = 60

    dp_g1 = ta_g1.get("_death_penalty", 0.0)
    dp_g5 = ta_g5.get("_death_penalty", 0.0)
    env_g1 = KAZWrapper(game_level=1, vector_state=True,
                        render_mode="rgb_array", seed=SEED,
                        death_penalty=dp_g1)
    env_g5 = KAZWrapper(game_level=5, vector_state=True,
                        render_mode="rgb_array", seed=SEED,
                        death_penalty=dp_g5)

    all_frames = []
    total_frames = 0

    for ep in range(EPISODES):
        obs1, _ = env_g1.reset()
        obs5, _ = env_g5.reset()
        flat1 = flatten(obs1)
        flat5 = flatten(obs5)
        ep_hidden5 = {}
        ep_stats = {"g1_kills": 0.0, "g5_kills": 0.0, "ep": ep + 1}

        while (env_g1.agents or env_g5.agents) and total_frames < MAX_FRAMES:
            # G1 actions
            if env_g1.agents:
                a1 = {}
                with torch.no_grad():
                    for ag in env_g1.agents:
                        obs_t = torch.tensor(flat1[ag], dtype=torch.float32,
                                             device=DEVICE).unsqueeze(0)
                        a1[ag] = int(net_g1.forward(obs_t)["logits"].argmax(-1).item())
                obs1, rew1, t1, tr1, info1 = env_g1.step(a1)
                flat1 = flatten(obs1) if env_g1.agents else {}
                for ag in rew1:
                    ep_stats["g1_kills"] += info1.get(ag, {}).get(
                        "reward_info", {}).get("raw_kill", 0.0)
            else:
                obs1, _ = env_g1.reset()
                flat1 = flatten(obs1)

            # G5 actions
            if env_g5.agents:
                a5 = {}
                with torch.no_grad():
                    for ag in env_g5.agents:
                        obs_t = torch.tensor(flat5[ag], dtype=torch.float32,
                                             device=DEVICE).unsqueeze(0)
                        h_np = ep_hidden5.get(ag)
                        h_t = (torch.tensor(h_np, dtype=torch.float32,
                                            device=DEVICE).unsqueeze(0)
                               if h_np is not None else None)
                        out = net_g5.forward(obs_t, hidden=h_t)
                        a5[ag] = int(out["logits"].argmax(-1).item())
                        if out["hidden"] is not None:
                            ep_hidden5[ag] = out["hidden"].detach().cpu().numpy()[0]
                obs5, rew5, t5, tr5, info5 = env_g5.step(a5)
                flat5 = flatten(obs5) if env_g5.agents else {}
                for ag in rew5:
                    ep_stats["g5_kills"] += info5.get(ag, {}).get(
                        "reward_info", {}).get("raw_kill", 0.0)
            else:
                obs5, _ = env_g5.reset()
                flat5 = flatten(obs5)
                ep_hidden5 = {}

            # Render both
            frame1 = env_g1.render()
            frame5 = env_g5.render()
            if frame1 is None or frame5 is None:
                continue

            # Resize to same height
            h = min(frame1.shape[0], frame5.shape[0])
            w = min(frame1.shape[1], frame5.shape[1])
            frame1 = cv2.resize(frame1, (w, h))
            frame5 = cv2.resize(frame5, (w, h))

            # Add text labels on each frame
            def add_label(frame, title, subtitle, kills):
                out_f = frame.copy()
                header = np.zeros((LABEL_H, w, 3), dtype=np.uint8)
                cv2.putText(header, title, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(header, f"{subtitle}  |  Kills: {kills:.0f}",
                            (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)
                return np.vstack([header, out_f])

            frame1_lab = add_label(frame1, "G1: GREEDY SOLDIER",
                                   "MLP  |  Infinite ammo, no fog",
                                   ep_stats["g1_kills"])
            frame5_lab = add_label(frame5, "G5: FIRE DISCIPLINE",
                                   "Attn+GRU  |  Fog + ammo + stamina",
                                   ep_stats["g5_kills"])

            # Add divider
            divider = np.full((h + LABEL_H, 4, 3), 200, dtype=np.uint8)
            combined = np.hstack([frame1_lab, divider, frame5_lab])
            all_frames.append(combined)
            total_frames += 1

    env_g1.close()
    env_g5.close()

    if not all_frames:
        print("  WARNING: no frames captured")
        return

    H_out, W_out = all_frames[0].shape[:2]
    out_path = "results/demo_sidebyside.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W_out, H_out))
    for frame in all_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  → {out_path}  ({len(all_frames)} frames, {len(all_frames)/FPS:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    part1_kill_density_evolution()
    part2_collapse_graph()
    part3_saliency()
    part4_sidebyside()
    print("\n[phase5] All artifacts saved to results/")
    print("  results/kill_density_evolution.png")
    print("  results/collapse_graph.png")
    print("  results/saliency_comparison.png")
    print("  results/demo_sidebyside.mp4")
