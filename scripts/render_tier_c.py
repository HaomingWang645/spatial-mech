#!/usr/bin/env python
"""Render Tier C perspective ego-videos.

Two modes:
  - From canonical 3D scenes (recommended):
        python scripts/render_tier_c.py --config configs/tier_c.yaml \
            --scenes-in data/scenes_3d --out data/tier_c \
            --trajectories-per-scene 2 --limit 100
  - Single-step (sample + render):
        python scripts/render_tier_c.py --config configs/tier_c.yaml \
            --out data/tier_c --n-scenes 100 --trajectories-per-scene 2
"""
import _bootstrap  # noqa: F401

from spatial_subspace.render.tier_c import main

if __name__ == "__main__":
    raise SystemExit(main())
