from typing import List, Dict, Optional, Union
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from collections import Counter
import matplotlib.pyplot as plt
import cv2
import json
import rasterio

from utils.tiff_utils import load_tif_image

from schema.bbox import BoundingBox
from schema.ClusterPolygon import ClusterPolygon

from algorithm.statistics import compute_average_leaf_width_per_cluster

class ClusterPalm:
    """
    Clustering palm tree bounding boxes into spatial groups with convex hull polygons.

    This class performs K-means clustering on palm tree bounding boxes (using center 
    coordinates + size features), filters small clusters, and generates convex hull 
    polygons for visualization and geospatial analysis. Supports both pixel and 
    georeferenced coordinate systems via TIFF metadata.

    Key Features:
        • K-means clustering with configurable cluster count
        • Minimum cluster size filtering 
        • Convex hull polygon generation per cluster
        • GeoJSON export with optional georeferencing
        • Leaf width computation per cluster
        • Interactive visualization with matplotlib

    Workflow:
        1. Initialize with n_clusters and min_cluster parameters
        2. Call predict(bbox_list, image) to cluster and generate polygons
        3. Access self.polygons: Dict[int, ClusterPolygon] with cluster results
        4. Export to GeoJSON or visualize with draw_cluster_polygons()

    Attributes
    ----------
    n_clusters : int
        Number of clusters to generate (default: 5)
    min_cluster : int  
        Minimum trees per cluster (default: 5)
    image : np.ndarray | None
        Reference image (RGB numpy array)
    canvas_h, canvas_w : int | None
        Image height and width in pixels
    bbox_list : List[BoundingBox] | None
        Input palm tree bounding boxes
    polygons : Dict[int, ClusterPolygon] | None
        Clustering results {cluster_id: ClusterPolygon}
    """
    def __init__(
        self,
        n_clusters: int = 5,
        min_cluster: int = 5
    ):
        """
        Initialize clustering configuration only.
        Actual image and bounding boxes are set by calling .predict().

        Parameters
        ----------
        n_clusters : int, default=5
            Target number of clusters for K-means
        min_cluster : int, default=5
            Discard clusters with fewer trees than this threshold
        """
        self.n_clusters = n_clusters
        self.min_cluster = min_cluster
        self.image = None
        self.canvas_h = None
        self.canvas_w = None
        self.bbox_list = None
        self.polygons = None

    def predict(
        self,
        bbox_list: List[BoundingBox],
        image: Union[str, np.ndarray]
    ):
        """
        Perform clustering and generate convex hull polygons.

        This method:
        1. Loads image (supports TIFF via load_tif_image or regular images)
        2. Extracts features: [center_x, center_y, width, height] per bbox
        3. Applies K-means clustering with size filtering
        4. Generates normalized convex hull polygons per cluster
        5. Stores results in self.polygons, self.bbox_list, self.image

        Parameters
        ----------
        bbox_list : List[BoundingBox]
            Palm tree detections from PalmDetector
        image : str or np.ndarray
            Image path (TIFF/JPG/PNG) or numpy array (HWC RGB)

        Returns
        -------
        Dict[int, ClusterPolygon]
            {cluster_id: ClusterPolygon(polygon, center, count)}

        Raises
        ------
        ValueError
            If image loading fails
        """
        if isinstance(image, str):
            if image.lower().endswith((".tif", ".tiff")):
                img = load_tif_image(image) 
            else:
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            img = np.array(image)

        self.image = img
        self.canvas_h, self.canvas_w = img.shape[0], img.shape[1]
        self.bbox_list = bbox_list
        self.polygons = self._cluster_polygons()

        self.polygons = self._cluster_polygons()

    def _cluster_polygons(self) -> Dict[int, ClusterPolygon]:
        """
        Cluster bounding boxes and generate polygons using current image and bbox_list.
        """
        if self.bbox_list is None or self.image is None:
            raise ValueError("Call fit() with bbox_list and image before clustering.")

        fitur = []
        centers = []
        for bbox in self.bbox_list:
            width = bbox.x2 - bbox.x1
            height = bbox.y2 - bbox.y1
            center_x = (bbox.x1 + bbox.x2) / 2
            center_y = (bbox.y1 + bbox.y2) / 2
            fitur.append([center_x, center_y, width, height])
            centers.append([center_x, center_y])

        X = np.array(fitur)
        centers = np.array(centers)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        labels = kmeans.fit_predict(X)
        count = Counter(labels)
        filtered_labels = np.array([l if count[l] >= self.min_cluster else -1 for l in labels])
        cx_min, cx_max = centers[:,0].min(), centers[:,0].max()
        cy_min, cy_max = centers[:,1].min(), centers[:,1].max()

        def norm(val, amin, amax, cmax):
            return (val - amin) / (amax - amin + 1e-8) * cmax
        norm_centers = np.column_stack((
            norm(centers[:,0], cx_min, cx_max, self.canvas_w),
            norm(centers[:,1], cy_min, cy_max, self.canvas_h),
        ))
        
        polygons = {}
        for cid in range(self.n_clusters):
            idxs = np.where(filtered_labels == cid)[0]
            if len(idxs) >= 3:
                pts = norm_centers[idxs]
                hull = ConvexHull(pts)
                poly_pts = pts[hull.vertices].tolist()
                mx, my = np.median(pts[:,0]), np.median(pts[:,1])
                polygons[cid] = ClusterPolygon(
                    polygon=poly_pts,
                    center=[mx, my],
                    count=len(idxs)
                )
        return polygons
    
    def compute_leaf_width_per_cluster(self) -> Dict[int, float]:
        """
        Compute average leaf/crown width per cluster in pixels.

        Uses external statistics function to map bboxes to clusters via 
        centroid distance and computes: 2 * mean(distance from bbox center to edges).

        Returns
        -------
        Dict[int, float]
            {cluster_id: average_leaf_width_pixels}

        Raises
        ------
        ValueError
            If predict() not called first (no polygons or bboxes)
        """
        if self.polygons is None or self.bbox_list is None:
            raise ValueError("Run predict() first.")
        return compute_average_leaf_width_per_cluster(self.polygons, self.bbox_list)

    def save_cluster_polygons_to_geojson(
        self, 
        tif_path: Optional[str],
        geojson_path: str, 
    ):
        """
        Export cluster polygons to GeoJSON FeatureCollection.

        Supports georeferencing if TIFF path provided:
        • Converts pixel (x,y) → geographic (lon,lat) using rasterio
        • Includes CRS metadata from TIFF
        • Each feature has cluster_id, center, count properties

        Parameters
        ----------
        tif_path : str, optional
            Georeferenced TIFF for pixel→geo transformation
        geojson_path : str, default="clusters.geojson"
            Output GeoJSON file path

        Raises
        ------
        ValueError
            If no polygons available
        rasterio.errors.RasterioIOError
            If TIFF file invalid
        """
        if self.polygons is None:
            raise ValueError("No polygons available. Run predict() first.")

        transform_func = None
        crs = None
        if tif_path is not None:
            with rasterio.open(tif_path) as src:
                transform = src.transform
                crs = src.crs
                def pixel_to_geo(x, y):
                    lon, lat = rasterio.transform.xy(transform, y, x)
                    return [lon, lat]
                transform_func = pixel_to_geo

        features = []
        for cid, cp in self.polygons.items():
            poly = cp.polygon
            coords = [transform_func(x, y) if transform_func else [x, y] for x, y in poly]
            # Close the ring as required by GeoJSON
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            props = {"cluster_id": cid, "center": cp.center, "count": cp.count}
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": props
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
        print(f"✅ Cluster polygons saved as GeoJSON in {geojson_path}")

    def draw_cluster_polygons(
        self,
        colors: Optional[List[str]] = None,
        show_label: bool = True,
        save_path: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ):
        """
        Visualize clusters overlaid on reference image.

        Creates matplotlib figure with:
        • Background image (self.image)
        • Semi-transparent cluster polygons
        • Cluster ID labels at polygon centers
        • Proper aspect ratio preserving image dimensions

        Parameters
        ----------
        colors : List[str], optional
            Polygon colors (cycles through list)
        show_label : bool, default=True
            Show cluster ID text labels
        save_path : str, optional
            Save figure to file (DPI=300)
        ax : plt.Axes, optional
            Existing axes for subplot integration
        """
        if self.polygons is None:
            raise ValueError("No polygons result available. \nRun .fit() first")
        polys = self.polygons

        if self.image is None:
            raise ValueError("No image available. \nRun .fit() first")
        img = self.image

        if img is None or polys is None:
            raise ValueError("Image and polygons must be set (call fit() first, or pass them explicitly).")

        canvas_h, canvas_w = img.shape[0], img.shape[1]
        if colors is None:
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'magenta', 'yellow', 'cyan']

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(canvas_w // 300, 8), max(canvas_h // 300, 8)))
            close_fig = True
        else:
            close_fig = False

        ax.set_xlim(0, canvas_w)
        ax.set_ylim(0, canvas_h)
        ax.set_facecolor('white')
        ax.imshow(img, extent=[0, canvas_w, 0, canvas_h])

        for i, (cid, cp) in enumerate(polys.items()):
            poly = cp.polygon
            center = cp.center
            color = colors[i % len(colors)]
            poly_patch = plt.Polygon(poly, closed=True, color=color, alpha=0.35)
            ax.add_patch(poly_patch)
            if show_label:
                ax.text(center[0], center[1], str(cid), color='white', fontsize=16, fontweight='bold',
                        ha='center', va='center', alpha=0.9)

        ax.set_title('Cluster polygons')
        ax.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if close_fig and not save_path:
            plt.show()
        if close_fig and save_path:
            plt.close()
