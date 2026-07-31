#!/bin/bash
# AFUN spike: run demo.py on each rendered LIBERO input (GT sim depth, no lingbot refinement).
set -e
SPIKE=/workspace/code/RoutedFlow/experiments/stage1_afun_spike
AFUN=/workspace/code/RoutedFlow/third_party/AFUN
PY=/workspace/miniconda3/envs/afun/bin/python
export HF_HOME=/workspace/.hf_home CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} AFUN_ROOT=$AFUN

cd "$AFUN"
for d in "$SPIKE"/inputs/*/; do
    task=$(basename "$d")
    query=$($PY -c "import json;print(json.load(open('$d/gt.json'))['query'])")
    echo "=== $task"
    $PY demo.py --rgb "$d/rgb.png" --query "$query" \
        --depth-dir "$d/depth" --no-refine --no-3d \
        --out "$SPIKE/outputs/$task"
done
echo "ALL SPIKE RUNS DONE"
