"""Natural-language QA generation from scene ground truth (plan §3.5).

Tier A covers relative position (left/right/front/behind) and distance
ordering. Containment and navigation are Tier B/C — they need volume
semantics and trajectories respectively.
"""
from __future__ import annotations

import random
from itertools import combinations

from ..scene import Object3D, QAItem


def _ref(o: Object3D) -> str:
    return f"the {o.color} {o.shape}"


def generate_qa(
    objects: list[Object3D],
    rng: random.Random,
    max_items: int = 10,
) -> list[QAItem]:
    items: list[QAItem] = []

    for a, b in combinations(objects, 2):
        dx = a.centroid[0] - b.centroid[0]
        dy = a.centroid[1] - b.centroid[1]
        if abs(dx) >= abs(dy):
            direction = "to the right of" if dx > 0 else "to the left of"
        else:
            direction = "in front of" if dy > 0 else "behind"
        items.append(
            QAItem(
                question=f"Is {_ref(a)} {direction} {_ref(b)}?",
                answer="yes",
                kind="relative_position",
                involves=[a.object_id, b.object_id],
            )
        )

    if len(objects) >= 3:
        for ref in objects:
            others = [o for o in objects if o.object_id != ref.object_id]
            a, b = rng.sample(others, 2)
            da = (a.centroid[0] - ref.centroid[0]) ** 2 + (a.centroid[1] - ref.centroid[1]) ** 2
            db = (b.centroid[0] - ref.centroid[0]) ** 2 + (b.centroid[1] - ref.centroid[1]) ** 2
            nearer = a if da < db else b
            items.append(
                QAItem(
                    question=f"Which is closer to {_ref(ref)}: {_ref(a)} or {_ref(b)}?",
                    answer=_ref(nearer),
                    kind="distance_order",
                    involves=[ref.object_id, a.object_id, b.object_id],
                )
            )

    rng.shuffle(items)
    return items[:max_items]
