#!/bin/bash
# Run MindCube eval on all 32 key checkpoints sequentially in batches of 4.
set -e

PY=/home/haoming/miniconda3/envs/vlm-ex/bin/python3
mkdir -p reports/mindcube_eval logs/v2/mindcube

eval_one() {
    local name=$1 mid=$2 gpu=$3
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu \
        $PY scripts/eval_mindcube.py \
        --model-id "$mid" \
        --checkpoint "checkpoints/${name}/lora" \
        --mindcube-jsonl /home/haoming/mindcube_data/raw/MindCube_tinybench.jsonl \
        --image-root /home/haoming/mindcube_data \
        --out "reports/mindcube_eval/${name}.json" \
        --n-eval 200 \
        > logs/v2/mindcube/${name}.log 2>&1
}

# Generate the 32-checkpoint list
declare -a CHECKPOINTS=()
for lam in 0 0.3 1 3.0; do
    for seed in 0 1 2 3; do
        CHECKPOINTS+=("qwen_lam${lam}_seed${seed}|Qwen/Qwen2.5-VL-7B-Instruct")
        CHECKPOINTS+=("intern_lam${lam}_seed${seed}|OpenGVLab/InternVL3-8B-hf")
    done
done

echo "[$(date +%H:%M:%S)] MindCube eval starting on ${#CHECKPOINTS[@]} checkpoints"

# Run in batches of 4
GPUS=(2 3 4 5)
i=0
for entry in "${CHECKPOINTS[@]}"; do
    name="${entry%%|*}"
    mid="${entry##*|}"
    gpu="${GPUS[$((i % 4))]}"
    eval_one "$name" "$mid" "$gpu" &
    i=$((i + 1))
    if [ $((i % 4)) -eq 0 ]; then
        wait
        n_done=$(ls reports/mindcube_eval/*.json 2>/dev/null | wc -l)
        echo "[$(date +%H:%M:%S)] batch $((i/4)) done; ${n_done} JSONs written"
    fi
done
wait
echo "[$(date +%H:%M:%S)] ALL_MINDCUBE_DONE"
