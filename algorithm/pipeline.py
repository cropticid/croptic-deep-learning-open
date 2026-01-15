import os
import numpy as np
import json
from typing import List

from algorithm.palmDetection import PalmDetector
from algorithm.cleanness import LandCleannessAnalyzer
from algorithm.clustering import ClusterPalm
from algorithm.infraDetection import InfraDetector

class palmAnalysisPipeline:
    def __init__(
        self,
        palm_detector_model_path: str,
        seg_model_path: str,
        config_path: str,
        grid_size: tuple = (30, 30),
        cluster_n: int = 5,
        min_cluster: int = 5,
        device: str = "cuda",
    ):
        self.grid_size = grid_size
        self.cluster_n = cluster_n
        self.min_cluster = min_cluster

        # Oil Palm Detector
        self.palm_detector = PalmDetector(palm_detector_model_path, config_path, device=device)
        
        # Infra Detector
        seg_model_name: str = seg_model_path
        selected_labels: List[str] = None
        self.seg_model = InfraDetector(model_name=seg_model_name, device=device)
        self.selected_labels = selected_labels or ["building"]

        # Cleanliness Analyzer
        self.cleanliness_analyzer = LandCleannessAnalyzer(grid_size=self.grid_size)

        # Cluster Palm
        self.clustering = ClusterPalm(n_clusters=self.cluster_n, min_cluster=self.min_cluster)

    def run(self, image_path: str, output_dir: str = "output"):
        os.makedirs(output_dir, exist_ok=True)

        # Detect bounding boxes
        bboxes = self.palm_detector.predict(image_path)
        output_bbox_path = os.path.join(output_dir, "bbox_palm.geojson")
        self.palm_detector.save_bboxes_to_geojson(image_path, output_bbox_path)
        #self.palm_detector.save_bboxes_to_json(output_json_path)
        #output_image_path = os.path.join(output_dir, "detection.png")
        #self.palm_detector.draw(output_image_path)
        
        # Segment selected region labels and extract polygons
        self.seg_model.predict_sliced(image_path, selected_labels=self.selected_labels)
        output_polygon_path = os.path.join(output_dir, "polygon_infra.geojson")
        self.seg_model.save_polygons_to_geojson(image_path, output_polygon_path)
        #with open(polygons_json_path, 'w') as f:
        #    json.dump(segmentation_result, f, indent=4)
        #print(f"✅ Saved polygons for labels {self.selected_labels} to {polygons_json_path}")
        
        # Analyze cleanness/greenness grid
        #cleanness_dir = os.path.join(output_dir, "cleanness")
        #os.makedirs(cleanness_dir, exist_ok=True)
        self.cleanliness_analyzer.predict(image_path, bboxes)
        #heatmap_path = os.path.join(output_dir, "cleanness_heatmap.png")
        #self.cleanliness_analyzer.save_cleanness_heatmap(save_path=heatmap_path)
        raster_path = os.path.join(output_dir, "cleanness_grid.tif")
        self.cleanliness_analyzer.save_cleanness_raster(output_tif_path=raster_path)

        # Step 4: Cluster boxes and draw clusters
        self.clustering.predict(bbox_list=bboxes, image=image_path)
        cluster_output_path = os.path.join(output_dir, "clusters_map.geojson")
        self.clustering.save_cluster_polygons_to_geojson(image_path, cluster_output_path)

        
