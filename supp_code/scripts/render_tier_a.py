#!/usr/bin/env python
"""Render Tier A scenes.

Example:
    python scripts/render_tier_a.py \
        --config configs/tier_a.yaml --out data/tier_a --n-scenes 5000
"""
import _bootstrap  # noqa: F401

from spatial_subspace.render.tier_a import main

if __name__ == "__main__":
    raise SystemExit(main())
