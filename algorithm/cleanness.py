import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from typing import Dict, List, Optional, Tuple

from utils.tiff_utils import load_tif_image

from schema.bbox import BoundingBox
from schema.GridCell import GridCell

class ColorAnalyzer:
    """
    Simple HSV-based vegetation (greenness) detector for image patches.

    Detects 'green' pixels using hue range [35°,85°] (green-yellow) and 
    saturation threshold for vegetation discrimination. Returns green pixel ratio.

    HSV Thresholds:
        - Hue: 35-85 (green vegetation range)
        - Saturation: ≥50% (excludes pale/grayscale)
    """
    def analyze_greenness(self, bgr_patch: np.ndarray) -> float:
        hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        h_mask = (hsv_patch[:, :, 0] >= 35) & (hsv_patch[:, :, 0] <= 85)
        s_mask = (hsv_patch[:, :, 1] >= 50)
        green_mask = (h_mask & s_mask)
        greenness = green_mask.sum() / (green_mask.size + 1e-8)
        return float(greenness)

class LandCleannessAnalyzer:
    """
    Grid-based land cleanness analysis excluding palm-covered areas.

    Computes vegetation greenness per grid cell, excluding regions with
    >70% palm coverage (auto-set cleanness=0.0).
    
    Outputs: cleanness matrix, GeoTIFF raster, overlay visualization.
    """

    def __init__(self, grid_size: Tuple[int, int] = (20, 20)):
        """
        Initialize grid-based cleanness analyzer.

        Parameters
        ----------
        grid_size : Tuple[int, int], default=(20, 20)
            Grid dimensions (rows, cols) for cleanness analysis.
            Higher resolution = finer granularity but slower processing.
        """
        self.grid_rows, self.grid_cols = grid_size
        self.color_analyzer = ColorAnalyzer()
        self.image = None
        self.image_path = None
        self.height = None
        self.width = None
        self.result = None

    def predict(self, image_path: str, palm_bboxes: List[BoundingBox] = None) -> Dict:
        """
        Analyze land cleanness across grid cells.

        1. Loads image (TIFF/RGB)
        2. Creates uniform grid overlay  
        3. Generates palm exclusion mask
        4. Computes greenness per cell (skip >70% palm)
        5. Stores matrix + stats in self.result

        Parameters
        ----------
        image_path : str
            TIFF/JPG/PNG image path
        palm_bboxes : List[BoundingBox], optional
            Palm detections to exclude from analysis

        Returns
        -------
        Dict
            {
                'mean_cleanness': float,
                'cleanness_matrix': np.ndarray(rows,cols), 
                'grid_cells': List[GridCell],
                'image_shape': Tuple[int,int]
            }

        Raises
        ------
        ValueError
            Cannot load image
        """
        self.image_path = image_path
        if image_path.lower().endswith((".tif", ".tiff")):
            self.image = load_tif_image(image_path)
        else:
            self.image = cv2.imread(image_path)
            if self.image is not None:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        if self.image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        self.height, self.width = self.image.shape[:2]
        grid_cells = self._create_grid(self.width, self.height)
        palm_mask = self._create_palm_mask(self.width, self.height, palm_bboxes or [])

        greenness_matrix = np.zeros((self.grid_rows, self.grid_cols))
        results = []

        for cell in grid_cells:
            patch = self.image[cell.y1:cell.y2, cell.x1:cell.x2]
            patch_mask = palm_mask[cell.y1:cell.y2, cell.x1:cell.x2]
            if patch.size == 0 or np.sum(patch_mask) / patch_mask.size > 0.7:
                cell.greenness = 0.0
                greenness_matrix[cell.row, cell.col] = 0.0
                continue
            greenness = self.color_analyzer.analyze_greenness(patch)
            cell.greenness = greenness
            greenness_matrix[cell.row, cell.col] = greenness
            results.append(cell)

        self.result = {
            'mean_cleanness': greenness_matrix.mean(),
            'cleanness_matrix': greenness_matrix,
            'grid_cells': results,
            'image_shape': (self.height, self.width)
        }
        return self.result

    def _create_grid(self, width: int, height: int) -> List[GridCell]:
        """
        Generate uniform rectangular grid cells covering image.

        Handles non-divisible dimensions (truncates edge cells).

        Parameters
        ----------
        width, height : int
            Image dimensions

        Returns
        -------
        List[GridCell]
            Grid cells with row/col/x1/y1/x2/y2
        """
        cells = []
        cell_width = width // self.grid_cols
        cell_height = height // self.grid_rows
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = min(x1 + cell_width, width)
                y2 = min(y1 + cell_height, height)
                cells.append(GridCell(row=row, col=col, x1=x1, y1=y1, x2=x2, y2=y2))
        return cells

    def _create_palm_mask(self, width: int, height: int, palm_bboxes: List[BoundingBox]) -> np.ndarray:
        """
        Binary mask marking palm-covered pixels.

        Parameters
        ----------
        width, height : int
            Image dimensions
        palm_bboxes : List[BoundingBox]
            Palm tree bounding boxes

        Returns
        -------
        np.ndarray
            (height,width) uint8 mask [0,1]
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        for bbox in palm_bboxes:
            mask[
                max(0, round(bbox.y1)) : min(round(bbox.y2), height),
                max(0, round(bbox.x1)) : min(round(bbox.x2), width)
            ] = 1
        return mask

    def save_cleanness_heatmap(self, result: Optional[Dict] = None, save_path: Optional[str] = None, alpha: float = 0.5):
        """
        Create and save RGB overlay visualization.

        Colormap: SUMMER (yellow=low → green=high cleanness)
        Blend: image * (1-alpha) + heatmap * alpha

        Parameters
        ----------
        result : Dict, optional
            Analysis result (uses self.result)
        save_path : str, optional
            Output PNG (DPI=300)
        alpha : float, default=0.5
            Heatmap overlay transparency
        """
        if result is None:
            if self.result is None or self.image is None:
                raise ValueError("No cleanness analysis result available. \nRun .fit() first")
            result = self.result

        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        matrix = result['cleanness_matrix']
        heatmap_resized = cv2.resize(matrix.astype(np.float32), (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * (heatmap_resized / heatmap_resized.max() if heatmap_resized.max() > 0 else 1))
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_SUMMER)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_color, alpha, 0)

        plt.figure(figsize=(12, 8))
        im = plt.imshow(overlay)
        plt.title(f"Overlay cleanness Heatmap\nMean greenness: {result['mean_cleanness']:.2f}")
        plt.axis('off')
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('cleanness (%)', fontsize=10)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #plt.show()

    def save_cleanness_raster(self, output_tif_path: str = "greenness_grid.tif", dtype='float32'):
        """
        Export cleanness matrix as georeferenced GeoTIFF.

        Preserves original image CRS/transform. Interpolates grid to image resolution.

        Parameters
        ----------
        output_tif_path : str, default="greenness_grid.tif"
            Output GeoTIFF path
        dtype : str, default='float32'
            Raster data type

        Raises
        ------
        ValueError
            No analysis result or image path
        rasterio.errors.RasterioIOError
            TIFF writing failed
        """
        if self.image_path is None:
            raise ValueError("No image path info available; run analyze_cleanness first.")

        with rasterio.open(self.image_path) as src:
            transform = src.transform
            crs = src.crs
            height, width = src.height, src.width

        # Use last result if none provided
        if self.result['cleanness_matrix'] is None:
            raise ValueError("No cleanness analysis result available. \nRun .fit() first")
        cleanness_matrix = self.result['cleanness_matrix']

        # Resize grid if needed
        if cleanness_matrix.shape != (height, width):
            greenness = cv2.resize(cleanness_matrix, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        else:
            greenness = cleanness_matrix

        with rasterio.open(
            output_tif_path,
            'w',
            driver='GTiff',
            height=greenness.shape[0],
            width=greenness.shape[1],
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(greenness, 1)
        print("GeoTIFF written to:", output_tif_path)


# Example Usage
if __name__ == "__main__":
    palm_boxes = [
        BoundingBox(100, 50, 200, 150),
        BoundingBox(300, 200, 450, 350),
    ]
    img_path = '/content/patch_165.jpg'
    analyzer = LandCleannessAnalyzer(grid_size=(30, 30))
    analyzer.analyze_cleanness(img_path, palm_boxes)
    analyzer.generate_greenness_heatmap(save_path='greenness_heatmap.png')
    analyzer.save_cleanness_raster(output_tif_path="greenness_grid.tif")
