#!/usr/bin/env bash
# Launch extraction jobs across GPUs for the topology Option 3 sweep.
# Each job is pinned to one GPU via CUDA_VISIBLE_DEVICES.
#
# Usage:
#   bash scripts/launch_extraction_sweep.sh qwen_7b_f32 2        # launch spec on GPU 2
#   bash scripts/launch_extraction_sweep.sh all                  # launch everything (needs 4 GPUs)
set -euo pipefail

cd /home/haoming/x-spatial-manual
PY=/home/haoming/miniconda3/envs/vlm-ex/bin/python3
mkdir -p logs

run() {
    local name=$1
    local data_root=$2
    local out_dir=$3
    local model_cfg=$4
    local gpu=$5

    local log=logs/extract_${name}.log
    echo "launching ${name} on GPU ${gpu} -> ${out_dir} (log=${log})"
    CUDA_VISIBLE_DEVICES=${gpu} nohup $PY scripts/extract_activations.py \
        --data-root "${data_root}" \
        --out-dir "${out_dir}" \
        --model-config "${model_cfg}" \
        --tier C --mode video \
        > "${log}" 2>&1 &
    echo "  pid=$!"
}

spec=${1:-all}
shift || true

case "${spec}" in
    qwen_7b_f32)
        gpu=${1:-2}
        run qwen_7b_f32 data/tier_c_free6dof_f32 \
            data/activations/tier_c_free6dof_f32_qwen25vl_7b \
            configs/models/qwen25vl.yaml "${gpu}"
        ;;
    qwen_7b_f64)
        gpu=${1:-3}
        run qwen_7b_f64 data/tier_c_free6dof_f64 \
            data/activations/tier_c_free6dof_f64_qwen25vl_7b \
            configs/models/qwen25vl.yaml "${gpu}"
        ;;
    llava_ov_f32)
        gpu=${1:-4}
        run llava_ov_f32 data/tier_c_free6dof_f32 \
            data/activations/tier_c_free6dof_f32_llava_ov_7b \
            configs/models/llava_ov_7b.yaml "${gpu}"
        ;;
    llava_ov_f64)
        gpu=${1:-5}
        run llava_ov_f64 data/tier_c_free6dof_f64 \
            data/activations/tier_c_free6dof_f64_llava_ov_7b \
            configs/models/llava_ov_7b.yaml "${gpu}"
        ;;
    internvl3_f16)
        gpu=${1:-2}
        run internvl3_f16 data/tier_c_free6dof \
            data/activations/tier_c_free6dof_internvl3_8b \
            configs/models/internvl3_8b.yaml "${gpu}"
        ;;
    internvl3_f32)
        gpu=${1:-3}
        run internvl3_f32 data/tier_c_free6dof_f32 \
            data/activations/tier_c_free6dof_f32_internvl3_8b \
            configs/models/internvl3_8b.yaml "${gpu}"
        ;;
    internvl3_f64)
        gpu=${1:-4}
        run internvl3_f64 data/tier_c_free6dof_f64 \
            data/activations/tier_c_free6dof_f64_internvl3_8b \
            configs/models/internvl3_8b.yaml "${gpu}"
        ;;
    round1)
        run qwen_7b_f32 data/tier_c_free6dof_f32 \
            data/activations/tier_c_free6dof_f32_qwen25vl_7b \
            configs/models/qwen25vl.yaml 2
        run qwen_7b_f64 data/tier_c_free6dof_f64 \
            data/activations/tier_c_free6dof_f64_qwen25vl_7b \
            configs/models/qwen25vl.yaml 3
        run llava_ov_f32 data/tier_c_free6dof_f32 \
            data/activations/tier_c_free6dof_f32_llava_ov_7b \
            configs/models/llava_ov_7b.yaml 4
        run llava_ov_f64 data/tier_c_free6dof_f64 \
            data/activations/tier_c_free6dof_f64_llava_ov_7b \
            configs/models/llava_ov_7b.yaml 5
        ;;
    round2)
        run internvl3_f16 data/tier_c_free6dof \
            data/activations/tier_c_free6dof_internvl3_8b \
            configs/models/internvl3_8b.yaml 2
        run internvl3_f32 data/tier_c_free6dof_f32 \
            data/activations/tier_c_free6dof_f32_internvl3_8b \
            configs/models/internvl3_8b.yaml 3
        run internvl3_f64 data/tier_c_free6dof_f64 \
            data/activations/tier_c_free6dof_f64_internvl3_8b \
            configs/models/internvl3_8b.yaml 4
        ;;
    *)
        echo "unknown spec: ${spec}"
        exit 1
        ;;
esac

echo "launched."
