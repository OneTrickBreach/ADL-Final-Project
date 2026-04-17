#!/bin/bash
# V3 one-shot training orchestration (plan §4.8).
# Runs heuristic baseline evals, selects best ammo mode, then trains G2→G5.
set -euo pipefail

mkdir -p models/v3 results/v3 results/tensorboard_v3
PY=./.venv/bin/python

echo "[1/7] G0 heuristic eval..."
$PY src/evaluate_v3.py --game g0 --episodes 10 --seed 42

echo "[2/7] G1a heuristic eval (global pool)..."
$PY src/evaluate_v3.py --game g1a --episodes 10 --seed 42

echo "[3/7] G1b heuristic eval (individual pool)..."
$PY src/evaluate_v3.py --game g1b --episodes 10 --seed 42

AMMO_MODE=$($PY -c "
import json
a = json.load(open('results/v3/g1a_eval_results.json'))
b = json.load(open('results/v3/g1b_eval_results.json'))
print('global' if a['score_mean'] >= b['score_mean'] else 'individual')
")
echo "  -> ammo_mode = $AMMO_MODE (used for G2-G5)"
echo "$AMMO_MODE" > results/v3/ammo_mode.txt

echo "[4/7] Training G2 (attention)..."
$PY src/train_v3.py --game g2 --ammo_mode "$AMMO_MODE" --total_timesteps 1000000

echo "[5/7] Training G3 (attn+GRU, <-G2)..."
$PY src/train_v3.py --game g3 --ammo_mode "$AMMO_MODE" --total_timesteps 1500000 \
    --transfer_from models/v3/g2/final.pt

echo "[6/7] Training G4 (+lock, <-G3)..."
$PY src/train_v3.py --game g4 --ammo_mode "$AMMO_MODE" --total_timesteps 1500000 \
    --transfer_from models/v3/g3/final.pt

echo "[7/7] Training G5 (+pragmatic, <-G4)..."
$PY src/train_v3.py --game g5 --ammo_mode "$AMMO_MODE" --total_timesteps 1500000 \
    --transfer_from models/v3/g4/final.pt

echo "V3 TRAINING COMPLETE. Tell Cascade to run evaluation pipeline."
