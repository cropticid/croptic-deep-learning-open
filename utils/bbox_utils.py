import json
from typing import Optional, List
from schema.bbox import BoundingBox, BBoxListInput

import json
from typing import List, Optional
from schema.bbox import BoundingBox, BBoxListInput


def labelme_json_to_bboxes(
    json_path: str,
    keep_labels: Optional[List[str]] = None,
) -> List[BoundingBox]:
    """
    Convert LabelMe JSON → list of BoundingBox (Pydantic)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    bboxes: List[BoundingBox] = []

    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        label = shape.get("label")

        if not points:
            continue

        if keep_labels and label not in keep_labels:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        bboxes.append(
            BoundingBox(
                x1=float(min(xs)),
                y1=float(min(ys)),
                x2=float(max(xs)),
                y2=float(max(ys)),
                label=label,
            )
        )

    return bboxes
