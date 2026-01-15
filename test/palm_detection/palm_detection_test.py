import sys
import os
import json
import numpy as np

# Add project root to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from algorithm.palmDetection import PalmDetector

model_path = "models/co_dino_5scale_swin_large_sawit_2024_pretrained_mopad/latest.pth"
config_path = "models/co_dino_5scale_swin_large_sawit_2024_pretrained_mopad/co_dino_5scale_swin_large_sawit_2024_pretrained_mopad.py"
test_img_path = "test/file/Lokasi 1 Lahan Sawet Patek.tif"

output_dir = "test/palm_detection"  

palm_detector = PalmDetector(model_path, config_path)

os.makedirs(output_dir, exist_ok=True)

# 1. Run detection, image and bboxes are stored in the class.
palm_detector.predict(test_img_path)

# 2. Draw overlays and save image (uses internal image/bboxes)
output_image_path = os.path.join(output_dir, "output_image_full.png")
print("Saving image to:", output_image_path)
palm_detector.draw(output_image_path)

# 3. Save bounding boxes to JSON (uses internal bboxes)
output_json_path = os.path.join(output_dir, "output_image_full.geojson")
print("Saving geojson to:", output_json_path)
palm_detector.save_bboxes_to_geojson(output_json_path)
