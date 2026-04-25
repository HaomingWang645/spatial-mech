#!/bin/bash
# Orchestrate the full Dirichlet-loss v2 experiment:
#  1. Wait for round 2 training (already launched)
#  2. Run OOD eval on all 8 round-1+2 checkpoints (4 GPUs parallel)
#  3. Aggregate; if VQA gain on Dirichlet < 1pp, launch lambda-sweep
#  4. Eval lambda-sweep checkpoints on both IID and OOD
#  5. Write aggregation file + plot
set -e

PY=/home/haoming/miniconda3/envs/vlm-ex/bin/python3
ROOT=/home/haoming/x-spatial-manual
cd $ROOT

mkdir -p logs/v2/eval reports/dirichlet_train_v2

# === Step 1: Wait for round 2 ===
echo "[$(date +%H:%M:%S)] Round 2 already complete (skipping wait)."

# === Step 2: OOD eval on 8 checkpoints (4 parallel batches) ===
echo "[$(date +%H:%M:%S)] Starting OOD eval on 8 checkpoints..."

eval_one() {
    local ckpt=$1 gpu=$2 name=$3 model_id=$4 val=$5 out=$6
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu \
        $PY scripts/eval_dirichlet_checkpoint.py \
        --model-id "$model_id" --checkpoint "$ckpt" \
        --val-jsonl "$val" --layer 17 --tau 2.0 \
        --n-eval 500 \
        --out "$out" \
        > logs/v2/eval/$name.log 2>&1
}
export -f eval_one
export PY

# Batch A: GPUs 2/3/4/5 — 4 parallel evals
declare -a CHECKPOINTS=(
    "qwen_lam0_seed0|Qwen/Qwen2.5-VL-7B-Instruct"
    "qwen_lam0_seed1|Qwen/Qwen2.5-VL-7B-Instruct"
    "qwen_lam1_seed0|Qwen/Qwen2.5-VL-7B-Instruct"
    "qwen_lam1_seed1|Qwen/Qwen2.5-VL-7B-Instruct"
    "intern_lam0_seed0|OpenGVLab/InternVL3-8B-hf"
    "intern_lam0_seed1|OpenGVLab/InternVL3-8B-hf"
    "intern_lam1_seed0|OpenGVLab/InternVL3-8B-hf"
    "intern_lam1_seed1|OpenGVLab/InternVL3-8B-hf"
)
GPUS=(2 3 4 5)
i=0
for entry in "${CHECKPOINTS[@]}"; do
    name="${entry%%|*}"
    model_id="${entry##*|}"
    gpu="${GPUS[$((i % 4))]}"
    eval_one \
        "checkpoints/${name}/lora" \
        "$gpu" \
        "${name}_ood" \
        "$model_id" \
        "data/dirichlet_train_v2/val_ood.jsonl" \
        "reports/dirichlet_train_v2/${name}_ood.json" &
    echo "  launched ${name}_ood on GPU $gpu (pid=$!)"
    i=$((i+1))
    # Run 4 in parallel; wait every 4
    if [ $((i % 4)) -eq 0 ]; then
        wait
        echo "[$(date +%H:%M:%S)] Eval batch $((i/4)) complete"
    fi
done
wait
echo "[$(date +%H:%M:%S)] All OOD evals done."

# === Step 3: Quick check — does Dirichlet beat baseline on OOD VQA? ===
echo "[$(date +%H:%M:%S)] Aggregating round-1+2 results..."
$PY scripts/aggregate_v2.py --out-prefix reports/dirichlet_train_v2/aggregate

# Decide whether to run lambda sweep based on the OOD VQA delta
delta=$($PY -c "
import json
def vqa(name): return json.load(open('reports/dirichlet_train_v2/' + name + '_ood.json'))['vqa_accuracy']
ql0 = (vqa('qwen_lam0_seed0') + vqa('qwen_lam0_seed1')) / 2
ql1 = (vqa('qwen_lam1_seed0') + vqa('qwen_lam1_seed1')) / 2
il0 = (vqa('intern_lam0_seed0') + vqa('intern_lam0_seed1')) / 2
il1 = (vqa('intern_lam1_seed0') + vqa('intern_lam1_seed1')) / 2
delta_q = ql1 - ql0
delta_i = il1 - il0
print(f'{(delta_q + delta_i) / 2:.4f}')
" 2>/dev/null || echo "0.0000")
echo "[$(date +%H:%M:%S)] Mean Dir-vs-baseline OOD VQA delta: $delta"

# Run lambda sweep if Dirichlet doesn't already win clearly on OOD
SWEEP_NEEDED=$(echo "$delta < 0.005" | bc -l 2>/dev/null || echo 1)

if [ "$SWEEP_NEEDED" = "1" ]; then
    echo "[$(date +%H:%M:%S)] OOD VQA delta below threshold — launching lambda sweep..."

    # Sweep: lambda ∈ {0.1, 0.3, 3.0} on Qwen seed 0 (4 GPUs parallel; one slot for lambda=0 control re-run)
    # Re-use lam0_seed0 baseline and lam1_seed0 from round 1; just add 3 more lambdas
    COMMON_ARGS="--train-jsonl data/dirichlet_train_v2/train.jsonl --val-jsonl data/dirichlet_train_v2/val_iid.jsonl --layer 17 --tau 2.0 --steps 500 --batch-size 2 --lora-rank 16 --eval-every 250 --log-every 50 --n-eval 100 --seed 0"

    for entry in "0.1|2" "0.3|3" "3.0|4"; do
        lam="${entry%%|*}"
        gpu="${entry##*|}"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/train_qwen_dirichlet.py \
            --model-id Qwen/Qwen2.5-VL-7B-Instruct $COMMON_ARGS \
            --output-dir checkpoints/qwen_lam${lam}_seed0 --lambda-dir $lam \
            > logs/v2/qwen_lam${lam}_seed0.log 2>&1 &
        echo "  GPU $gpu lambda=$lam PID=$!"
    done
    # GPU 5: also do an InternVL lambda sweep at 0.3 (most promising mid-strength)
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 nohup $PY scripts/train_qwen_dirichlet.py \
        --model-id OpenGVLab/InternVL3-8B-hf $COMMON_ARGS \
        --output-dir checkpoints/intern_lam0.3_seed0 --lambda-dir 0.3 \
        > logs/v2/intern_lam0.3_seed0.log 2>&1 &
    echo "  GPU 5 intern lambda=0.3 PID=$!"

    wait
    echo "[$(date +%H:%M:%S)] Lambda sweep training done. Eval-ing IID + OOD..."

    # Eval all 4 new checkpoints
    declare -a SWEEP_NEW=(
        "qwen_lam0.1_seed0|Qwen/Qwen2.5-VL-7B-Instruct"
        "qwen_lam0.3_seed0|Qwen/Qwen2.5-VL-7B-Instruct"
        "qwen_lam3.0_seed0|Qwen/Qwen2.5-VL-7B-Instruct"
        "intern_lam0.3_seed0|OpenGVLab/InternVL3-8B-hf"
    )
    i=0
    for entry in "${SWEEP_NEW[@]}"; do
        name="${entry%%|*}"
        model_id="${entry##*|}"
        gpu="${GPUS[$i]}"
        # OOD eval
        eval_one "checkpoints/${name}/lora" "$gpu" "${name}_ood" "$model_id" \
            "data/dirichlet_train_v2/val_ood.jsonl" \
            "reports/dirichlet_train_v2/${name}_ood.json" &
        i=$((i+1))
    done
    wait
    echo "[$(date +%H:%M:%S)] Sweep OOD evals done."

    # IID eval the new ones too (their FINAL log already had IID, but we want unified output)
    i=0
    for entry in "${SWEEP_NEW[@]}"; do
        name="${entry%%|*}"
        model_id="${entry##*|}"
        gpu="${GPUS[$i]}"
        eval_one "checkpoints/${name}/lora" "$gpu" "${name}_iid" "$model_id" \
            "data/dirichlet_train_v2/val_iid.jsonl" \
            "reports/dirichlet_train_v2/${name}_iid.json" &
        i=$((i+1))
    done
    wait
fi

# === Step 4: Final aggregation ===
echo "[$(date +%H:%M:%S)] Final aggregation..."
$PY scripts/aggregate_v2.py --out-prefix reports/dirichlet_train_v2/aggregate
echo "[$(date +%H:%M:%S)] DONE."
