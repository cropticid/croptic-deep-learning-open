import sys
import os

# Add project root to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from algorithm.cleanness import LandCleannessAnalyzer
from schema.bbox import BoundingBox
from utils.bbox_utils import labelme_json_to_bboxes

# The image path must be to your image file (not JSON)
img_path = 'test/file/Lokasi 2 Lahan Sawet Patek.tif'  

# Your palm_boxes should be a list of BoundingBox objects, or leave as empty list
palm_boxes = labelme_json_to_bboxes("test/file/merge_output.json")

# Create analyzer with the grid size only
analyzer = LandCleannessAnalyzer(grid_size=(30, 30))

# Analyze palm cleanliness; result is now stored in analyzer.result
analyzer.predict(img_path, palm_boxes)

# Plot and save heatmap (uses latest analysis result)
analyzer.save_cleanness_heatmap(save_path='test\cleanness\cleanness_heatmap.png')

# Save result as GeoTIFF raster
analyzer.save_cleanness_raster(output_tif_path="test\cleanness\cleanness_grid.tif")
