#!/bin/bash
# V3.1 one-shot training orchestration (plan docs/v3_1_retrain_plan_fixed.md §6).
# Runs heuristic baseline evals under the V3.1 detector + pressure params,
# selects best ammo mode, then trains G2→G5. Outputs isolated under
# models/v3_1/, results/v3_1/, results/tensorboard_v3_1/ so V3 artifacts
# remain intact for the sensitivity appendix.
set -euo pipefail

mkdir -p models/v3_1 results/v3_1 results/tensorboard_v3_1
PY=./.venv/bin/python
RESULTS=results/v3_1
SUF=_1

echo "[1/7] G0 heuristic eval..."
$PY src/evaluate_v3.py --game g0 --episodes 10 --seed 42 --results_dir "$RESULTS"

echo "[2/7] G1a heuristic eval (global pool)..."
$PY src/evaluate_v3.py --game g1a --episodes 10 --seed 42 --results_dir "$RESULTS"

echo "[3/7] G1b heuristic eval (individual pool)..."
$PY src/evaluate_v3.py --game g1b --episodes 10 --seed 42 --results_dir "$RESULTS"

AMMO_MODE=$($PY -c "
import json
a = json.load(open('$RESULTS/g1a_eval_results.json'))
b = json.load(open('$RESULTS/g1b_eval_results.json'))
print('global' if a['score_mean'] >= b['score_mean'] else 'individual')
")
echo "  -> ammo_mode = $AMMO_MODE (used for G2-G5)"
echo "$AMMO_MODE" > "$RESULTS/ammo_mode.txt"

echo "[4/7] Training G2 (attention)..."
$PY src/train_v3.py --game g2 --ammo_mode "$AMMO_MODE" \
    --total_timesteps 1000000 --output_suffix "$SUF"

echo "[5/7] Training G3 (attn+GRU, <-G2)..."
$PY src/train_v3.py --game g3 --ammo_mode "$AMMO_MODE" \
    --total_timesteps 1500000 --output_suffix "$SUF" \
    --transfer_from models/v3_1/g2/final.pt

echo "[6/7] Training G4 (+lock, <-G3)..."
$PY src/train_v3.py --game g4 --ammo_mode "$AMMO_MODE" \
    --total_timesteps 1500000 --output_suffix "$SUF" \
    --transfer_from models/v3_1/g3/final.pt

echo "[7/7] Training G5 (+pragmatic, <-G4)..."
$PY src/train_v3.py --game g5 --ammo_mode "$AMMO_MODE" \
    --total_timesteps 2000000 --output_suffix "$SUF" \
    --transfer_from models/v3_1/g4/final.pt

echo "V3.1 TRAINING COMPLETE. Tell Cascade to run evaluation pipeline."
