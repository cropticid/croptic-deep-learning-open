# file: palm_cli.py
import os
import time
import argparse
from algorithm.pipeline import palmAnalysisPipeline


def run_pipeline_local(
    input_dir: str,
    output_root: str,
    palm_detector_model_path: str,
    seg_model_path: str,
    palm_detector_config_path: str,
    grid_size=(30, 30),
    cluster_n: int = 7,
    min_cluster: int = 7,
    device: str = "cuda",
):
    os.makedirs(output_root, exist_ok=True)

    tif_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".tif", ".tiff"))
    ]

    if not tif_paths:
        print("No .tif files found in input_dir")
        return

    start = time.time()

    pipeline = palmAnalysisPipeline(
        palm_detector_model_path=palm_detector_model_path,
        seg_model_path=seg_model_path,
        config_path=palm_detector_config_path,
        grid_size=grid_size,
        cluster_n=cluster_n,
        min_cluster=min_cluster,
        device=device,
    )

    for tif_path in tif_paths:
        filename = os.path.basename(tif_path)
        stem = os.path.splitext(filename)[0]

        tif_output_dir = os.path.join(output_root, stem)
        os.makedirs(tif_output_dir, exist_ok=True)

        print(f"[INFO] Processing {tif_path} -> {tif_output_dir}")
        pipeline.run(tif_path, tif_output_dir)

    print(f"[DONE] Runtime: {time.time() - start:.2f} sec")


def main():
    parser = argparse.ArgumentParser(description="Palm analysis local inference")
    parser.add_argument("--input-dir", required=True, help="Folder berisi file .tif lokal")
    parser.add_argument("--output-dir", required=True, help="Folder output")
    parser.add_argument("--palm-detector-model", required=True, help="Path model detector palm")
    parser.add_argument("--seg-model", required=True, help="Path model segmentation")
    parser.add_argument("--palm-detector-config-path", required=True, help="Path config pipeline/model")
    parser.add_argument("--device", default="cuda", help="cuda atau cpu")
    parser.add_argument("--cluster-n", type=int, default=7)
    parser.add_argument("--min-cluster", type=int, default=7)
    parser.add_argument("--grid-size", type=int, nargs=2, default=[30, 30])

    args = parser.parse_args()

    run_pipeline_local(
        input_dir=args.input_dir,
        output_root=args.output_dir,
        palm_detector_model_path=args.palm_detector_model,
        seg_model_path=args.seg_model,
        palm_detector_config_path=args.palm_detector_config_path,
        grid_size=tuple(args.grid_size),
        cluster_n=args.cluster_n,
        min_cluster=args.min_cluster,
        device=args.device,
    )


if __name__ == "__main__":
    main()
