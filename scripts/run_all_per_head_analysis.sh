#!/usr/bin/env bash
# Orchestrate per-head probing + visualizations once extractions finish.
# Usage: bash scripts/run_all_per_head_analysis.sh {7b,32b,all}

set -euo pipefail

which_model="${1:-all}"

run_for () {
    model="$1"               # "7b" or "32b"
    activ="data/activations/tier_c_free6dof_qwen25vl_${model}_per_head"
    probes="data/probes/tier_c_free6dof/qwen25vl_${model}_per_head_camera_depth"
    full_probe="data/probes/tier_c_free6dof/qwen25vl_${model}_camera_depth/camera_depth_probes.json"
    figs="figures/tier_c_free6dof_per_head/qwen25vl_${model}"
    mkdir -p "$probes" "$figs" data/probes/tier_c_free6dof/qwen25vl_${model}_per_head_cumulative data/probes/tier_c_free6dof/qwen25vl_${model}_per_head_spec

    echo "=============================="
    echo "Per-head analysis: ${model}"
    echo "=============================="

    echo "[1/3] fit per (layer, head) probes"
    title_model="Qwen2.5-VL-${model^^}"   # "7B" / "32B"
    /home/haoming/miniconda3/envs/vlm-ex/bin/python scripts/fit_probes_camera_depth_per_head.py \
        --activations "$activ" \
        --scenes data/tier_c_free6dof \
        --out "$probes" \
        --title-model "$title_model"
    cp "$probes"/heatmaps.png "$figs"/
    cp "$probes"/heatmaps_components.png "$figs"/

    echo "[2/3] cumulative top-k + attention-vs-full-layer"
    /home/haoming/miniconda3/envs/vlm-ex/bin/python scripts/analyze_per_head_cumulative.py \
        --per-head "$probes" \
        --activations "$activ" \
        --scenes data/tier_c_free6dof \
        --out "data/probes/tier_c_free6dof/qwen25vl_${model}_per_head_cumulative" \
        ${full_probe:+--full-layer-probe $full_probe}
    cp "data/probes/tier_c_free6dof/qwen25vl_${model}_per_head_cumulative"/cumulative.png "$figs"/ || true
    cp "data/probes/tier_c_free6dof/qwen25vl_${model}_per_head_cumulative"/attention_vs_fulllayer.png "$figs"/ 2>/dev/null || true

    echo "[3/3] specialization"
    /home/haoming/miniconda3/envs/vlm-ex/bin/python scripts/analyze_head_specialization.py \
        --per-head "$probes" \
        --out "data/probes/tier_c_free6dof/qwen25vl_${model}_per_head_spec"
    cp "data/probes/tier_c_free6dof/qwen25vl_${model}_per_head_spec"/specialization.png "$figs"/
    cp "data/probes/tier_c_free6dof/qwen25vl_${model}_per_head_spec"/head_r2_matrix.png "$figs"/

    echo "done: ${model}  figures in ${figs}"
}

if [ "$which_model" = "7b" ] || [ "$which_model" = "all" ]; then
    run_for 7b
fi
if [ "$which_model" = "32b" ] || [ "$which_model" = "all" ]; then
    run_for 32b
fi
