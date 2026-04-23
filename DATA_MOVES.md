# Data moves — 2026-04-22

`/home` hit 100% full while running the per-head attribution experiment. The eight directories below were moved from `/home/haoming/x-spatial-manual/data/activations/` to `/mnt/data3/haoming_x_spatial_scratch/` and replaced with **absolute symlinks** so every script / path / report reference continues to resolve unchanged.

All moves used `mv` (cross-partition, so it's a full copy + unlink); the source device (`/dev/nvme2n1`, mounted at `/home`) was the full one, the destination (`/dev/nvme4n1p1`, mounted at `/mnt/data3`) has ample space.

## Moved directories

| Directory | Size | Age at move | Reason |
|---|---|---|---|
| `tier_a_qwen25vl_7b/` | 2.1 GB | 9 days | Stale — original Apr 13 tier A probe run, not touched since |
| `tier_b_qwen25vl_7b/` | 1.4 GB | 9 days | Stale — original Apr 13 tier B probe run |
| `tier_c_qwen25vl_7b/` | 3.2 GB | 9 days | Stale — original Apr 13 tier C orbit probe run |
| `tier_c_free6dof_qwen25vl_7b_per_head/` | 84 GB | 0 days (today) | Re-creatable in ~60 min via `extract_per_head.py`; probe metrics already saved |
| `tier_c_free6dof_qwen25vl_32b_per_head/` | 73 GB | 0 days (today) | Re-creatable in ~34 min; probe metrics already saved |
| `tier_c_free6dof_qwen25vl_72b/` | 39 GB | 1 day | Not touched in 24 h; full-layer activation store |
| `tier_c_free6dof_qwen25vl_32b/` | 20 GB | 1 day | Not touched in 24 h; full-layer activation store |
| `tier_c_free6dof_llava_ov_7b/` | 12 GB | 1 day | Not touched in 24 h; full-layer activation store |
| `tier_c_free6dof_f32_llava_ov_7b/` | 12 GB | 0 d | Second round — llava-ov activation cleanup |
| `tier_c_free6dof_f64_llava_ov_7b/` | 24 GB | 0 d | Second round |
| `tier_c_free6dof_f8_llava_ov_7b/` | 3.0 GB | 0 d | Second round |
| `tier_d_llava_ov_7b/` | 614 MB | 0 d | Second round — Tier D llava-ov |
| `tier_d_7scenes_llava_ov_7b/` | 528 MB | 0 d | Second round |
| `tier_d_kitti_llava_ov_7b/` | 614 MB | 0 d | Second round |

| `pw_n16_internvl3_38b/` | 2.2 GB | 0 d | Third round — `pw_n*` cleanup |
| `pw_n16_qwen25vl_32b/` | 1.5 GB | 0 d | Third round |
| `pw_n16_qwen25vl_7b/` | 445 MB | 0 d | Third round |
| `pw_n32_internvl3_38b/` | 3.6 GB | 0 d | Third round |
| `pw_n32_qwen25vl_32b/` | 2.4 GB | 0 d | Third round |
| `pw_n32_qwen25vl_7b/` | 725 MB | 0 d | Third round |
| `pw_n64_internvl3_38b/` | 4 KB (empty) | 0 d | Third round |
| `pw_n64_qwen25vl_32b/` | 3.3 GB | 0 d | Third round |
| `pw_n64_qwen25vl_7b/` | 1.0 GB | 0 d | Third round |

**Total moved: ~290 GB across 23 directories.**

## Symlinks created

At each original path `/home/haoming/x-spatial-manual/data/activations/<name>/`, an absolute symlink now points to `/mnt/data3/haoming_x_spatial_scratch/<name>/`. Verified working by `pd.read_parquet` + `np.load(..., mmap_mode='r')` through the symlinks.

```
data/activations/tier_a_qwen25vl_7b                      → /mnt/data3/haoming_x_spatial_scratch/tier_a_qwen25vl_7b
data/activations/tier_b_qwen25vl_7b                      → /mnt/data3/haoming_x_spatial_scratch/tier_b_qwen25vl_7b
data/activations/tier_c_qwen25vl_7b                      → /mnt/data3/haoming_x_spatial_scratch/tier_c_qwen25vl_7b
data/activations/tier_c_free6dof_qwen25vl_7b_per_head    → /mnt/data3/haoming_x_spatial_scratch/tier_c_free6dof_qwen25vl_7b_per_head
data/activations/tier_c_free6dof_qwen25vl_32b_per_head   → /mnt/data3/haoming_x_spatial_scratch/tier_c_free6dof_qwen25vl_32b_per_head
data/activations/tier_c_free6dof_qwen25vl_72b            → /mnt/data3/haoming_x_spatial_scratch/tier_c_free6dof_qwen25vl_72b
data/activations/tier_c_free6dof_qwen25vl_32b            → /mnt/data3/haoming_x_spatial_scratch/tier_c_free6dof_qwen25vl_32b
data/activations/tier_c_free6dof_llava_ov_7b             → /mnt/data3/haoming_x_spatial_scratch/tier_c_free6dof_llava_ov_7b
data/activations/tier_c_free6dof_f32_llava_ov_7b         → /mnt/data3/haoming_x_spatial_scratch/tier_c_free6dof_f32_llava_ov_7b
data/activations/tier_c_free6dof_f64_llava_ov_7b         → /mnt/data3/haoming_x_spatial_scratch/tier_c_free6dof_f64_llava_ov_7b
data/activations/tier_c_free6dof_f8_llava_ov_7b          → /mnt/data3/haoming_x_spatial_scratch/tier_c_free6dof_f8_llava_ov_7b
data/activations/tier_d_llava_ov_7b                      → /mnt/data3/haoming_x_spatial_scratch/tier_d_llava_ov_7b
data/activations/tier_d_7scenes_llava_ov_7b              → /mnt/data3/haoming_x_spatial_scratch/tier_d_7scenes_llava_ov_7b
data/activations/tier_d_kitti_llava_ov_7b                → /mnt/data3/haoming_x_spatial_scratch/tier_d_kitti_llava_ov_7b
```

## Disk state over time

|  | `/home` (source) | `/mnt/data3` (dest) |
|---|---|---|
| Before any move | **0 B free** / 100% used | 1.3 TB free / 62% used |
| After round 1 (8 dirs, 234.7 GB) | 234 GB free / 94% used | 1.1 TB free / 69% used |
| After round 2 (+6 llava dirs, ~40 GB) | 271 GB free / 92% used | 1001 GB free / 71% used |
| After deletes (SD backup + llava + 72B HF cache) | **460 GB free / 87% used** | 1001 GB free / 71% used |
| After round 3 (+9 `pw_n*` dirs, ~15 GB) | **475 GB free / 86% used** | **987 GB free / 71% used** |

## Restoring a moved directory in place

If any of these needs to live back on `/home` (e.g. because `/mnt/data3` is being reclaimed), the reverse is:

```bash
name=tier_c_free6dof_qwen25vl_72b
rm /home/haoming/x-spatial-manual/data/activations/$name           # removes the symlink only
mv /mnt/data3/haoming_x_spatial_scratch/$name \
   /home/haoming/x-spatial-manual/data/activations/$name
```

## Things I did NOT touch

- `tier_c_free6dof_internvl3_38b/` (41 GB, 1 day old) — left alone on `/home`. Proposed for moving but not confirmed.
- All Apr 22 (today) full-layer activation stores except the per-head fp16 tensors — left alone because they are this-session outputs that are easy to inspect locally.
- Everything outside `data/activations/`.
