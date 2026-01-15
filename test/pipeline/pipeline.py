import sys
import os
import time

# Add project root to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from algorithm.pipeline import palmAnalysisPipeline

model_path = "models/co_dino_5scale_swin_large_sawit_2024_pretrained_mopad/latest.pth"
config_path = "models/co_dino_5scale_swin_large_sawit_2024_pretrained_mopad/co_dino_5scale_swin_large_sawit_2024_pretrained_mopad.py"
img_path = "test/file/Lokasi 1 Lahan Sawet Patek.tif"
output_dir = "test/pipeline"

start_time = time.time()

pipeline = palmAnalysisPipeline(model_path, config_path, cluster_n=7, min_cluster=7)
results = pipeline.run(img_path, output_dir)

end_time = time.time()
elapsed_seconds = end_time - start_time

print("Finished pipeline!")
print(f"Total runtime: {elapsed_seconds:.2f} seconds")

# Add optional further analyses here, using results as needed
