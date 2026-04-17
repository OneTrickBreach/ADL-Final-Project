"""V3 artifact generation: figures + side-by-side demo.

Generates:
  - results/v3/score_evolution.png
  - results/v3/metric_breakdown.png
  - results/v3/saliency_v3.png        (best-effort, skipped gracefully if missing)
  - results/v3/demo_sidebyside_v3.mp4 (best-effort from G0 and G5 videos)

Assumes evaluate_v3.py has been run for all 7 games.
"""
import json
import os
import sys
from glob import glob

import numpy as np

sys.path.insert(0, ".")


GAMES = ["g0", "g1a", "g1b", "g2", "g3", "g4", "g5"]


def _load(game, det=False):
    suffix = "_det" if det else ""
    path = f"results/v3/{game}_eval_results{suffix}.json"
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def fig_score_evolution():
    import matplotlib.pyplot as plt
    data = {g: _load(g) for g in GAMES}
    means = [d["score_mean"] if d else 0 for d in data.values()]
    stds = [d["score_std"] if d else 0 for d in data.values()]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#888"] * 3 + ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    bars = ax.bar(GAMES, means, yerr=stds, color=colors, capsize=4)
    ax.set_ylabel("Score (kills − failures)")
    ax.set_title("V3 Score Evolution G0 → G5")
    ax.axhline(0, color="k", linewidth=0.5)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = "results/v3/score_evolution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[fig] {out}")


def fig_metric_breakdown():
    import matplotlib.pyplot as plt
    data = {g: _load(g) for g in GAMES}
    games = [g for g in GAMES if data[g] is not None]
    def _vals(key):
        return [data[g][key] for g in games]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].bar(games, _vals("ammo_pct_expended_mean"), color="#4c72b0")
    axes[0, 0].set_title("% Ammo expended")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 1].bar(games, _vals("stamina_pct_expended_mean"), color="#55a868")
    axes[0, 1].set_title("% Stamina expended")
    axes[0, 1].set_ylim(0, 1.05)
    # Kills A vs K
    ax = axes[1, 0]
    x = np.arange(len(games))
    ax.bar(x - 0.2, _vals("kills_archer_mean"), 0.4, label="Archer", color="#4c72b0")
    ax.bar(x + 0.2, _vals("kills_knight_mean"), 0.4, label="Knight", color="#c44e52")
    ax.set_xticks(x); ax.set_xticklabels(games)
    ax.set_title("Kills (archer vs knight)"); ax.legend()
    # Attacks A vs K
    ax = axes[1, 1]
    ax.bar(x - 0.2, _vals("attacks_archer_mean"), 0.4, label="Archer", color="#4c72b0")
    ax.bar(x + 0.2, _vals("attacks_knight_mean"), 0.4, label="Knight", color="#c44e52")
    ax.set_xticks(x); ax.set_xticklabels(games)
    ax.set_title("Attacks (archer vs knight)"); ax.legend()
    plt.tight_layout()
    out = "results/v3/metric_breakdown.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[fig] {out}")


def fig_saliency_v3():
    """Attention heatmaps for G3 vs G5 (best-effort)."""
    try:
        import matplotlib.pyplot as plt
        import torch
        from src.models.mappo_net import MAPPONet
        from src.wrappers.kaz_wrapper_v3 import KAZWrapperV3, EXTRA_DIM
        from src.utils import get_device
    except Exception as e:
        print(f"[fig] saliency skipped ({e})")
        return

    device = get_device()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for i, g in enumerate(["g3", "g5"]):
        ax = axes[i]
        ckpt_path = f"models/v3/{g}/final.pt"
        if not os.path.isfile(ckpt_path):
            ax.set_title(f"{g} (no ckpt)"); ax.axis("off"); continue
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        ta = ck["args"]
        arch = ck.get("arch", "attention_gru")
        env = KAZWrapperV3(game_level=g, ammo_mode_override=ta.get("ammo_mode", "individual"),
                           seed=42)
        obs, _ = env.reset(seed=42)
        obs_flat = {a: o.flatten().astype(np.float32) for a, o in obs.items()}
        obs_dim = list(obs_flat.values())[0].shape[0]
        act_dim = env.action_space(env.agents[0]).n
        net = MAPPONet(obs_dim=obs_dim, act_dim=act_dim,
                       hidden_dim=ta.get("hidden_dim", 256), arch=arch,
                       num_entities=27, entity_dim=5, num_heads=4,
                       extra_dim=EXTRA_DIM).to(device)
        net.load_state_dict(ck["model_state_dict"])
        net.eval()
        # Step a few to populate zombies
        for _ in range(20):
            if not env.agents: break
            env.step({a: 5 for a in env.agents})
        if not env.agents:
            ax.set_title(f"{g} (done)"); ax.axis("off"); env.close(); continue
        obs_raw, _ = env.reset(seed=42)
        for _ in range(30):
            obs_raw, _, _, _, _ = env.step({a: 5 for a in env.agents})
            if not env.agents: break
        agent = env.agents[0]
        ob = obs_raw[agent].flatten().astype(np.float32)
        obs_t = torch.tensor(ob, device=device).unsqueeze(0)
        # Capture attention weights via hook
        attn_weights = {}
        def hook(module, inp, out):
            # out: (attn_output, attn_weights)
            attn_weights["w"] = out[1].detach().cpu().numpy()
        h = net.entity_attention.attention.register_forward_hook(hook)
        with torch.no_grad():
            net.forward(obs_t)
        h.remove()
        w = attn_weights.get("w")
        if w is not None:
            im = ax.imshow(w[0], cmap="viridis", aspect="auto")
            ax.set_title(f"{g}: attention (self=0)")
            plt.colorbar(im, ax=ax, fraction=0.04)
        env.close()
    plt.tight_layout()
    out = "results/v3/saliency_v3.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[fig] {out}")


def demo_sidebyside():
    """Stitch G0 and G5 demo videos side-by-side (best-effort)."""
    try:
        import cv2
    except Exception:
        print("[demo] cv2 missing, skipping")
        return
    def _find(game):
        cand = sorted(glob(f"results/v3/{game}_demo/episode_*.mp4"))
        return cand[0] if cand else None
    a = _find("g0"); b = _find("g5")
    if not (a and b):
        print(f"[demo] need both g0 ({a}) and g5 ({b}); skipping")
        return
    va = cv2.VideoCapture(a); vb = cv2.VideoCapture(b)
    fps = va.get(cv2.CAP_PROP_FPS) or 15
    wa = int(va.get(cv2.CAP_PROP_FRAME_WIDTH)); ha = int(va.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wb = int(vb.get(cv2.CAP_PROP_FRAME_WIDTH)); hb = int(vb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    H = max(ha, hb)
    W = wa + wb
    out = "results/v3/demo_sidebyside_v3.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid = cv2.VideoWriter(out, fourcc, fps, (W, H))
    while True:
        ra, fa = va.read(); rb, fb = vb.read()
        if not (ra and rb): break
        pad_a = cv2.copyMakeBorder(fa, 0, H - ha, 0, 0, cv2.BORDER_CONSTANT)
        pad_b = cv2.copyMakeBorder(fb, 0, H - hb, 0, 0, cv2.BORDER_CONSTANT)
        vid.write(np.concatenate([pad_a, pad_b], axis=1))
    va.release(); vb.release(); vid.release()
    print(f"[demo] {out}")


def ablation_table():
    rows = []
    for g in GAMES:
        d = _load(g)
        if d is None:
            rows.append((g, "—", "—", "—", "—", "—", "—", "—"))
            continue
        rows.append((
            g, f"{d['score_mean']:.2f} ± {d['score_std']:.2f}",
            f"{d['ammo_pct_expended_mean']:.2f}",
            f"{d['stamina_pct_expended_mean']:.2f}",
            f"{d['kills_archer_mean']:.1f} / {d['kills_knight_mean']:.1f}",
            f"{d['attacks_archer_mean']:.1f} / {d['attacks_knight_mean']:.1f}",
            f"{d['failures_mean']:.1f}",
            f"{d['episode_length_mean']:.0f}",
        ))
    hdr = ("Game", "Score", "%Ammo", "%Stam", "Kills A/K", "Attacks A/K",
           "Failures", "EpLen")
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    out = "results/v3/ablation_table.md"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[table] {out}")


if __name__ == "__main__":
    os.makedirs("results/v3", exist_ok=True)
    fig_score_evolution()
    fig_metric_breakdown()
    fig_saliency_v3()
    demo_sidebyside()
    ablation_table()
