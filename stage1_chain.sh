#!/bin/bash
# Stage-1 queue 2026-07-31: 5-fold rotation (fold0 done) + no-prior ablation + evals.
set -e
cd /workspace/code/RoutedFlow
for fold in 1 2 3 4; do
    python3 run_stage1.py train-l1 --fold $fold --epochs 200
done
python3 run_stage1.py train-l1 --fold 0 --epochs 200 --no-prior
for run in fold0_seed0 fold1_seed0 fold2_seed0 fold3_seed0 fold4_seed0 fold0_seed0_noprior; do
    python3 run_stage1.py eval-l1 --run $run
done
echo "STAGE1 CHAIN DONE"
