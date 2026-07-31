#!/bin/bash
# Overnight chain 2026-07-30: stage-0 retrain on REPAIRED masks + n=400 evals.
# Old (mask-bug) runs archived at experiments/stage0_routing_causal_test/runs_invalid_maskbug/
set -e
cd /workspace/code/RoutedFlow
for mode in object_only robot_only phase_switched; do
    python3 run_stage0.py train --mode $mode
done
for mode in object_only robot_only phase_switched; do
    python3 run_stage0.py eval --mode $mode --nroll 40
done
echo "NIGHT CHAIN DONE"
