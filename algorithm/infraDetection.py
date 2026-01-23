import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.measure import regionprops
from PIL import Image
import json
import rasterio

from sahi.slicing import slice_image
from sahi.utils.cv import read_image_as_pil

import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from typing import Optional, Union, List, Dict
import cv2
import os

class InfraDetector:
    """
    Semantic segmentation for infrastructure detection using HuggingFace Transformers.

    Supports large images via SAHI-style sliced prediction with overlap stitching.
    Extracts vector polygons from raster masks using OpenCV contours. Detects 
    classes like 'building', 'road', etc. (model-dependent). Provides GeoJSON 
    export with georeferencing support.

    Key Features:
        • Transformers + SAHI slicing (512×512 tiles, 20% overlap)
        • Confidence-based filtering (≥0.9 threshold)
        • Pixel → GeoJSON polygon conversion  
        • Georeferenced export via TIFF metadata
        • Interactive visualization with label overlays
        • Model-agnostic (auto-discovers id2label classes)

    Workflow:
        1. Initialize(model_name="nvidia/segformer-b0-finetuned-ade-512-512")
        2. predict_sliced(image, ["building"]) for large images
        3. save_polygons_to_geojson(image.tif, output.geojson)
        4. show_segmented() for visualization
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Load semantic segmentation model from HuggingFace.

        Parameters
        ----------
        model_name : str
            HF model ID (e.g. "nvidia/segformer-b0-finetuned-ade-512-512")
        device : str, default="cuda"
            "cuda" or "cpu"
        """
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        self.last_predicted_mask: Optional[np.ndarray] = None
        self.last_image: Optional[Image.Image] = None
        self.last_predicted_polygons: Optional[Dict[str, List[List[List[float]]]]] = None

    def preprocess(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """
        Load and standardize image to RGB PIL format.

        Parameters
        ----------
        image_input : str or PIL.Image
            Image path or PIL image

        Returns
        -------
        PIL.Image
            RGB PIL image (stored as self.last_image)
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")
        self.last_image = image
        return image

    def mask_to_polygons(self, mask: np.ndarray, label_id: int) -> List[List[List[float]]]:
        """
        Extract polygon contours from binary mask using OpenCV.

        Simplifies contours with CHAIN_APPROX_SIMPLE. Filters tiny contours.

        Parameters
        ----------
        mask : np.ndarray
            Semantic segmentation mask (H,W) int32
        label_id : int
            Class ID to extract

        Returns
        -------
        List[List[List[float]]]
            List of polygon rings: [[[x1,y1], [x2,y2], ...]]
        """
        binary = (mask == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            pts = cnt.squeeze(axis=1)
            if pts.ndim == 1:
                pts = pts[np.newaxis, :]
            polygons.append([[float(x), float(y)] for x, y in pts])
        return polygons

    def predict(self, image_input: Union[str, Image.Image], selected_labels: List[str]) -> Dict[str, List[List[List[float]]]]:
        """
        Single-image semantic segmentation (non-sliced).

        For small images (<512×512). Returns label→polygons mapping.

        Parameters
        ----------
        image_input : str or PIL.Image
            Input image
        selected_labels : List[str]
            Model class names to extract (e.g. ["building"])

        Returns
        -------
        Dict[str, List[List[List[float]]]]
            {label_name: [polygons]}

        Raises
        ------
        ValueError
            Unknown label name
        """
        image = self.preprocess(image_input)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        self.last_predicted_mask = mask

        id2label = self.model.config.id2label
        label_to_id = {v: k for k, v in id2label.items()}

        output = {}
        for label in selected_labels:
            if label not in label_to_id:
                raise ValueError(f"Label '{label}' not found in model classes: {list(id2label.values())}")
            label_id = label_to_id[label]
            polygons = self.mask_to_polygons(mask, label_id)
            output[label] = polygons
        
        self.last_predicted_polygons = output 
        return output  # Dict: {label: [polygons]}

    def predict_sliced(
        self,
        image_input: Union[str, Image.Image],
        selected_labels: List[str],
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_ratio: float = 0.2,
    ):
        """
        High-resolution segmentation via sliced prediction.

        SAHI-style: slices → predict → stitch → confidence filter → polygons.
        Overlap averaging prevents edge artifacts.

        Parameters
        ----------
        image_input : str or PIL.Image
            Large input image (drone/aerial)
        selected_labels : List[str]
            Classes to extract polygons for
        slice_height, slice_width : int, default=512
            Tile size for processing
        overlap_ratio : float, default=0.2
            Tile overlap fraction

        Returns
        -------
        Dict[str, List[List[List[float]]]]
            {label: [polygons]} from full-resolution mask

        Raises
        ------
        ValueError
            Unknown labels
        """
        # Load original image
        image = self.preprocess(image_input)
        original_w, original_h = image.size

        # Slice into overlapping tiles
        slices = slice_image(
            image=image,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
        )

        full_mask = np.zeros((original_h, original_w), dtype=np.int32)
        count_mask = np.zeros((original_h, original_w), dtype=np.int32)

        # Predict on each slice and stitch back
        for s in slices:   # Each "s" is the dict from SliceImageResult
            tile = s["image"]   # numpy array
            x1, y1 = s["starting_pixel"]  # (left, top)
            h, w = tile.shape[:2]

            x2 = x1 + w
            y2 = y1 + h

            # Run SegFormer on tile
            inputs = self.processor(images=Image.fromarray(tile), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
                conf = torch.max(probs, dim=1)[0].squeeze().cpu().numpy()

            # Filter mask based on confidence treshold
            confidence_threshold = 0.9   # adjust as needed
            background_index = 0
            masked_mask = np.where(conf >= confidence_threshold, mask, background_index)

            if masked_mask.shape != (h, w):
                mask_img = Image.fromarray(masked_mask.astype(np.uint8))
                mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
                masked_mask = np.array(mask_img)

            full_mask[y1:y2, x1:x2] += masked_mask
            count_mask[y1:y2, x1:x2] += (masked_mask != background_index)

        # Normalize overlapping regions
        full_mask = full_mask / np.maximum(count_mask, 1)
        full_mask = full_mask.astype(np.int32)

        # Save to class
        self.last_predicted_mask = full_mask

        # Extract polygons only for selected labels
        id2label = self.model.config.id2label
        label_to_id = {v: k for k, v in id2label.items()}

        output = {}
        for label in selected_labels:
            if label not in label_to_id:
                raise ValueError(f"Label '{label}' not found in model classes.")
            label_id = label_to_id[label]
            polygons = self.mask_to_polygons(full_mask, label_id)
            output[label] = polygons

        self.last_predicted_polygons = output
        return output

    @property
    def available_labels(self) -> List[str]:
        """
        Get all class names supported by current model.

        Returns
        -------
        List[str]
            Model id2label values
        """
        return list(self.model.config.id2label.values())

    def get_labels_in_mask(self) -> List[int]:
        """
        Detect which classes appear in last_predicted_mask.

        Returns
        -------
        List[int]
            Present class IDs (pixel values >0)
        """
        present = np.unique(self.last_predicted_mask)
        return [int(i) for i in present if np.sum(self.last_predicted_mask == i) > 0]

    def save_polygons_to_geojson(
        self,
        tif_path: str,
        geojson_path
    ):
        """
        Export last_predicted_polygons as GeoJSON FeatureCollection.

        Optional pixel→geographic transformation using TIFF metadata.

        Parameters
        ----------
        tif_path : str, optional
            Georeference source (pixel→lon/lat)
        geojson_path : str
            Output GeoJSON file

        Raises
        ------
        ValueError
            No polygons available
        """
        label_polygon_dict = self.last_predicted_polygons
        if label_polygon_dict is None:
            raise ValueError("No polygon data available. Run predict() first.")

        transform_func = None
        crs = None
        if tif_path is not None:
            import rasterio
            with rasterio.open(tif_path) as src:
                transform = src.transform
                crs = src.crs
                def pixel_to_geo(x, y):
                    lon, lat = rasterio.transform.xy(transform, y, x)
                    return [lon, lat]
                transform_func = pixel_to_geo

        features = []
        for label, polygons in label_polygon_dict.items():
            for poly in polygons:
                coords = [transform_func(x, y) if transform_func else [x, y] for x, y in poly]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    },
                    "properties": {"label": label}
                }
                features.append(feature)
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        if crs is not None:
            geojson["crs"] = {"type": "name", "properties": {"name": str(crs)}}
        with open(geojson_path, "w") as f:
            json.dump(geojson, f, indent=2)
        print(f"✅ Segmentation polygons saved as GeoJSON in {geojson_path}")

    def show_segmented(
        self,
        output_image_path: Optional[str] = None,
        legend=True,
        overlay_alpha=0.5,
        figsize=(15, 8)
    ):
        """
        Interactive visualization: original | segmentation overlay.

        Features:
        • Random distinct colors per class
        • Centroid label text overlays  
        • Legend for detected classes only
        • Side-by-side comparison

        Parameters
        ----------
        output_image_path : str, optional
            Save PNG
        legend : bool, default=True
            Show class legend
        overlay_alpha : float, default=0.5
            Transparency
        figsize : tuple, default=(15,8)
            Figure size
        """
        if self.last_image is None or self.last_predicted_mask is None:
            raise ValueError("Run predict first.")
        id2label = self.model.config.id2label
        num_labels = len(id2label)
        colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
        cmap = ListedColormap(colors / 255.0)
        labels_in_image = self.get_labels_in_mask()

        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(self.last_image)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Segmentation with Labels")
        plt.imshow(self.last_image)
        plt.imshow(self.last_predicted_mask, cmap=cmap, alpha=overlay_alpha)

        # Overlay label text on regions
        for label_id in labels_in_image:
            mask_binary = (self.last_predicted_mask == label_id).astype(np.uint8)
            props = regionprops(mask_binary)
            label_name = id2label[label_id]
            for prop in props:
                y, x = prop.centroid
                plt.text(x, y, label_name, color='white', fontsize=10, fontweight='bold',
                         ha='center', va='center', bbox=dict(facecolor=colors[label_id]/255.0, alpha=0.7, pad=1))

        # Add legend for seen labels
        if legend:
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=id2label[i],
                                  markerfacecolor=colors[i]/255.0, markersize=10) for i in labels_in_image]
            plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.axis("off")
        plt.tight_layout()
        if output_image_path:
            plt.savefig(output_image_path, bbox_inches='tight')
            print(f"✅ Segmentation image saved in {output_image_path}")
        plt.show()


