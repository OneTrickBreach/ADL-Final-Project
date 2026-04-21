"""Re-encode every MP4 under results/v3_1/ at half playback speed.

Each output frame is duplicated `factor` times while keeping the container FPS
identical, which doubles wall-clock duration without dropping any information.
Originals are preserved — slowed copies land in ``results/v3_1_slow/`` with
the same relative layout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def slowdown(src: Path, dst: Path, factor: int) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"cannot write {dst}")
    n_written = 0
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        for _ in range(factor):
            out.write(frm)
            n_written += 1
    cap.release()
    out.release()
    return (n_in / fps, n_written / fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", default="results/v3_1")
    ap.add_argument("--dst_root", default="results/v3_1_slow")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--pattern", default="**/*.mp4")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    vids = sorted(src_root.glob(args.pattern))
    print(f"Found {len(vids)} videos under {src_root}")
    for v in vids:
        rel = v.relative_to(src_root)
        dst = dst_root / rel
        try:
            in_s, out_s = slowdown(v, dst, args.factor)
            print(f"  {rel}  {in_s:.1f}s -> {out_s:.1f}s")
        except Exception as exc:
            print(f"  {rel}  FAILED: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
