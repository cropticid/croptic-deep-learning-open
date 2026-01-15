import numpy as np
import cv2
import os
import json
import rasterio
from typing import List, Optional, Union
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from utils.tiff_utils import load_tif_image
from schema.bbox import BoundingBox

class_colors = {
    'sawit': (0, 255, 0),
    'sawit muda': (0, 0, 255),
    'sawit abnormal': (255, 0, 0)
}

class PalmDetector:
    def __init__(self,
                 model_path: str,
                 config_path: str,
                 confidence_threshold: float = 0.4,
                 image_size: int = 640,
                 device: str = "cuda"):
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='mmdet',
            model_path=model_path,
            config_path=config_path,
            confidence_threshold=confidence_threshold,
            image_size=image_size,
            device=device
        )
        self.image_np: Optional[np.ndarray] = None  
        self.bboxes: Optional[List[BoundingBox]] = None 

    def predict(self, image: Union[str, np.ndarray]) -> List[BoundingBox]:
        # Load image if input is a file path
        if isinstance(image, str):
            if image.lower().endswith((".tif", ".tiff")):
                img = load_tif_image(image)
            else:
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise ValueError("image must be a file path or numpy array")
        self.image_np = img  # Save to class variable

        # Detect and save
        self.bboxes = self.detect()
        return self.bboxes

    def detect(self) -> List[BoundingBox]:
        if self.image_np is None:
            raise ValueError("Image not loaded. Call fit(image) first.")
        result = get_sliced_prediction(
            self.image_np,
            self.detection_model,
            slice_height=1024,
            slice_width=1024,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        palm_bboxes: List[BoundingBox] = []
        for pred in result.object_prediction_list:
            x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
            label = pred.category.name
            palm_bboxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label))
        self.bboxes = palm_bboxes  # Save to class variable
        return palm_bboxes

    def draw(self, output_image_path: Optional[str] = None):
        if self.image_np is None:
            raise ValueError("No image loaded. Use fit() first.")
        if self.bboxes is None:
            raise ValueError("No bounding boxes available. Run fit() first.")
        image = self.image_np.copy()
        for bbox in self.bboxes:
            cx = int((bbox.x1 + bbox.x2) // 2)
            cy = int((bbox.y1 + bbox.y2) // 2)
            label = bbox.label
            if label != "sawit":
                color = class_colors.get(label, (128, 128, 128))
                cv2.circle(image, (cx, cy), radius=70, color=color, thickness=-1)
        if output_image_path:
            cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"✅ Gambar hasil disimpan di {output_image_path}")
        return image

    def save_bboxes_to_json(self, save_path: str):
        if self.bboxes is None:
            raise ValueError("No bounding boxes available. Run fit() first.")
        dicts = [bbox.dict() for bbox in self.bboxes]
        with open(save_path, 'w') as f:
            json.dump(dicts, f, indent=4)
        print(f"✅ Bounding box disimpan di {save_path}")
    
    def save_bboxes_to_geojson(self, tif_path: str, geojson_path: str):
        """
        Save bounding boxes as GeoJSON polygons using georeferenced coordinates from TIFF.
        """
        if self.bboxes is None:
            raise ValueError("No bounding boxes available. Run predict() first.")
        with rasterio.open(tif_path) as src:
            transform = src.transform
            crs = src.crs

            def pixel_to_geo(x, y):
                lon, lat = rasterio.transform.xy(transform, y, x)
                return [lon, lat]

            features = []
            for bbox in self.bboxes:
                x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                coords = [
                    pixel_to_geo(x1, y1),
                    pixel_to_geo(x2, y1),
                    pixel_to_geo(x2, y2),
                    pixel_to_geo(x1, y2),
                    pixel_to_geo(x1, y1)  # close ring
                ]
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    },
                    "properties": {"label": bbox.label}
                }
                features.append(feature)
            geojson = {
                "type": "FeatureCollection",
                "features": features,
                "crs": {"type": "name", "properties": {"name": str(crs)}}
            }
            with open(geojson_path, "w") as f:
                json.dump(geojson, f, indent=2)
            print(f"✅ Bounding boxes saved as GeoJSON in {geojson_path}")