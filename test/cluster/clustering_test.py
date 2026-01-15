import sys
import os
import json
import numpy as np

# Add project root to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from schema.bbox import BoundingBox
from algorithm.clustering import ClusterPalm

# Load bounding boxes
with open('test/file/output_image_full.json', 'r') as f:
    bbox_dicts = json.load(f)
bbox_list = [BoundingBox(**bb) for bb in bbox_dicts]

# Use a real image or white canvas
canvas_w, canvas_h = 3000, 3000
white_img = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

# 1. Create the ClusterPalm object (no data yet)
clustering = ClusterPalm(n_clusters=5, min_cluster=5)

# 2. Fit with data (clustering is performed here)
clustering.predict(bbox_list=bbox_list, image=white_img)

# 3. Draw and save the clusters (uses stored image and polygons)
save_path = "test/cluster/clusters_map.png"
clustering.draw_cluster_polygons(show_label=True, save_path=save_path)
