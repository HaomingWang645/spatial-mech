# Paper Code

Python source for the NeurIPS 2026 submission
**"Uncovering and Shaping the Latent Representation of Scene Topology in VLMs for Spatial Understanding"**
([../neurips_2026_main.tex](../neurips_2026_main.tex)).

This folder contains only `.py` files extracted from the project tree. Configuration
files (`configs/*.yaml`), shell launchers (`*.sh`), data, and checkpoints are kept
in the parent repository and are not duplicated here.

## Layout

```
paper_code/
├── scripts/   # Entry-point scripts (91 files): data generation, training, eval, plotting
├── src/       # spatial_subspace/ package (18 files) imported by the scripts
└── tests/     # Unit tests (9 files)
```

## Files cited directly in the paper

| Section | File | Purpose |
|---|---|---|
| Appendix (synthetic-scene pipeline) | [scripts/generate_scenes.py](scripts/generate_scenes.py) | Sample canonical 3D scenes (`scene.json`). |
| Appendix (synthetic-scene pipeline) | [scripts/render_tier_c.py](scripts/render_tier_c.py) | Render egocentric 16-frame videos under orbit / Free6DoF cameras. |
| Appendix (synthetic-scene pipeline) | [scripts/build_dirichlet_train_data.py](scripts/build_dirichlet_train_data.py) | Emit `train.jsonl` / `val.jsonl` for Dirichlet fine-tuning. |
| Appendix (training) | [scripts/train_with_dirichlet.py](scripts/train_with_dirichlet.py) | End-to-end LoRA + Dirichlet training loop. |
| Appendix (training) | [scripts/dirichlet_loss.py](scripts/dirichlet_loss.py) | Dirichlet-energy objective applied via residual-stream forward hook. |

## Functional groupings (other scripts)

- **Scene generation / rendering** — `generate_scenes.py`, `render_tier_{a,b,c}.py`,
  `render_tier_c_frame_sweep.py`, `generate_person_walk_dataset.py`,
  `visualize_scenes_3d_bev.py`.
- **Activation extraction & probing** — `extract_activations.py`,
  `extract_per_head.py`, `extract_lora_features.py`, `extract_vsi_frames.py`,
  `probe_color_shape_position.py`, `probe_temporal_dynamics.py`,
  `fit_probes_q1.py`, `fit_probes_camera_depth*.py`, `cross_trajectory_probe.py`,
  `build_residualization_basis.py`.
- **Topology analysis** — `topology_option3.py`, `topology_option3_residual.py`,
  `analyze_head_specialization.py`, `analyze_per_head_cumulative.py`,
  `shortcut_depth_analysis.py`.
- **Dirichlet training & ablations** — `train_with_dirichlet.py`,
  `train_qwen_dirichlet{,_v2}.py`, `train_qwen_spatID.py`, `dirichlet_pilot.py`,
  `eval_dirichlet_checkpoint.py`.
- **Real-world / counterfactual pilots** — `realworld_method2_pilot.py`,
  `realworld_method2_v2.py`, `realworld_method3_pilot.py`,
  `realworld_method3_v2.py`, `realworld_counterfactual_pilot.py`,
  `method_d_counterfactual.py`, `eval_counterfactuals.py`.
- **Benchmarks** — `eval_vsi.py`, `eval_vsi_batched.py`, `eval_vsi_generation.py`,
  `eval_vsi_with_ablation.py`, `eval_mindcube.py`, `eval_viewspatial.py`,
  `eval_ost_bench.py`, `eval_cam_motion_lora.py`, `cam_motion_vqa.py`,
  `depth_vqa.py`, `build_vsi_eval_data.py`, `run_vsi_full_queue.py`.
- **Activation steering** — `activation_steering.py`,
  `activation_steering_text.py`, `activation_steering_text_multi.py`.
- **Sweep orchestration** — `build_queue*.py`, `run_experiment_queue.py`,
  `aggregate_v2.py`, `aggregate_phase10_13.py`, `rescue_analysis.py`.
- **Figure rendering** — `draw_fig*.py`, `plot_*.py`, `visualize_*.py`.
- **Library** — [src/spatial_subspace/](src/spatial_subspace/) provides
  `extract`, `scene`, `probes`, `labels`, `metrics`, `models`, `datasets`,
  `utils`, and a `render/` subpackage.

## Notes

- Scripts assume the repository root as the working directory and import the
  `spatial_subspace` package from `src/`. Set `PYTHONPATH` accordingly, or run
  the scripts from the parent directory of this folder.
- Data paths (`data/`, `checkpoints/`, `figures/`) and YAML configs
  (`configs/tier_c_*.yaml`) referenced by these scripts are **not** in this
  folder; they live in the parent repository.
- Hardware: a single 80 GB H100 is sufficient to reproduce the 500-step
  Dirichlet runs (≈40–60 min/run, ≈55 GB peak activation memory at bf16).
