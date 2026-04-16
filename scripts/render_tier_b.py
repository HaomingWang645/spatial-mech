#!/usr/bin/env python
"""Render Tier B fragmented-BEV videos.

Two modes:
  - From canonical 3D scenes (recommended):
        python scripts/render_tier_b.py --config configs/tier_b.yaml \
            --scenes-in data/scenes_3d --out data/tier_b
  - Single-step (sample + render):
        python scripts/render_tier_b.py --config configs/tier_b.yaml \
            --out data/tier_b --n-scenes 200
"""
import _bootstrap  # noqa: F401

from spatial_subspace.render.tier_b import main

if __name__ == "__main__":
    raise SystemExit(main())
