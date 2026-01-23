import os
import numpy as np
import json
from typing import List

from algorithm.palmDetection import PalmDetector
from algorithm.cleanness import LandCleannessAnalyzer
from algorithm.clustering import ClusterPalm
from algorithm.infraDetection import InfraDetector

class palmAnalysisPipeline:
    """
    End-to-end oil palm plantation analysis pipeline.

    Orchestrates four core analysis modules:
    1. PalmDetector → Bounding boxes (normal/young/abnormal)
    2. InfraDetector → Infrastructure polygons (buildings, etc.)
    3. LandCleannessAnalyzer → Vegetation greenness grid (excl. palms)
    4. ClusterPalm → Spatial clustering + convex hull polygons

    Outputs 4 GeoJSON/TIFF files:
    - bbox_palm.geojson: Palm detections
    - polygon_infra.geojson: Buildings/roads  
    - cleanness_grid.tif: Greenness raster
    - clusters_map.geojson: Palm clusters

    Designed for drone/aerial satellite imagery of oil palm plantations.
    """

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
        """
        Initialize full analysis pipeline with all sub-modules.

        Parameters
        ----------
        palm_detector_model_path : str
            Palm detection model weights (.pth)
        seg_model_path : str  
            Segmentation model HF ID (e.g. "nvidia/segformer...")
        config_path : str
            MMDetection config (.py) for palm detector
        grid_size : tuple[int,int], default=(30,30)
            Cleanness analysis grid (rows,cols)
        cluster_n : int, default=5
            Target palm cluster count
        min_cluster : int, default=5
            Discard clusters with <N palms
        device : str, default="cuda"
            "cuda" or "cpu" for all models
        """
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
        """
        Execute complete analysis pipeline on single image.

        **Processing Steps** (sequential):
        1. **Palm Detection** → bbox_palm.geojson
        2. **Infrastructure Segmentation** → polygon_infra.geojson  
        3. **Land Cleanness Grid** → cleanness_grid.tif
        4. **Palm Clustering** → clusters_map.geojson

        Creates `output_dir/` automatically.

        Parameters
        ----------
        image_path : str
            Input image (TIFF/JPG/PNG, georeferenced preferred)
        output_dir : str, default="output"
            Output directory

        Returns
        -------
        Dict[str,str]
            Output file paths mapping

        Raises
        ------
        FileNotFoundError
            Model paths invalid
        ValueError
            Image loading/model inference failures
        """
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
        #leaf_widths = clustering.compute_leaf_width_per_cluster() # Compute averange leaf width
        cluster_output_path = os.path.join(output_dir, "clusters_map.geojson")
        self.clustering.save_cluster_polygons_to_geojson(image_path, cluster_output_path)

        
