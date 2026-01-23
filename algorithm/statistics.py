from typing import Dict, List, Tuple
import numpy as np

from schema.ClusterPolygon import ClusterPolygon
from schema.bbox import BoundingBox


def compute_average_leaf_width_per_cluster(polygons: Dict[int, ClusterPolygon], 
                                          bbox_list: List[BoundingBox]) -> Dict[int, float]:
    """
    Compute average leaf/crown width PER CLUSTER.

    How it works:
    1. Map bboxes to cluster labels via centroid distance
    2. Per cluster: compute 2 * mean(distance from bbox center to edges)
    3. Return dict {cluster_id: avg_leaf_width_pixels}

    Parameters
    ----------
    polygons : Dict[int, ClusterPolygon]
        Result from self.polygons (after predict())
    bbox_list : List[BoundingBox]
        Complete bounding box list

    Returns
    -------
    Dict[int, float]
        {cluster_id: avg_leaf_width_pixels}
    """
    if not bbox_list or not polygons:
        return {}
    
    # Buat mapping bbox ke cluster terdekat
    cluster_assignments = {}
    bbox_centers = [(i, (b.x1+b.x2)/2, (b.y1+b.y2)/2) for i, b in enumerate(bbox_list)]
    
    for cid, cluster in polygons.items():
        cluster_center = np.array(cluster.center)
        distances = []
        
        for idx, cx, cy in bbox_centers:
            bbox_center = np.array([cx, cy])
            dist = np.linalg.norm(cluster_center - bbox_center)
            distances.append((dist, idx))
        
        # Ambil bbox terdekat untuk cluster ini
        closest_idx = min(distances, key=lambda x: x[0])[1]
        cluster_assignments[cid] = closest_idx
    
    # Hitung avg leaf width per cluster
    leaf_widths = {}
    for cid, bbox_idx in cluster_assignments.items():
        bbox = bbox_list[bbox_idx]
        cx, cy = (bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0
        
        # Jarak center ke 4 sisi
        dx_left = cx - bbox.x1
        dx_right = bbox.x2 - cx
        dy_top = cy - bbox.y1
        dy_bottom = bbox.y2 - cy
        
        avg_radius = np.mean([dx_left, dx_right, dy_top, dy_bottom])
        leaf_width = 2.0 * avg_radius
        leaf_widths[cid] = float(leaf_width)
    
    return leaf_widths
