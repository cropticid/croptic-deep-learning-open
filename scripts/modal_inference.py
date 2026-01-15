import modal
import os
import time
import boto3

app = modal.App("croptic-deeplearning")

aws_secret = modal.Secret.from_name(
    "aws-secret"  
)

image = ( modal.Image.from_aws_ecr(
    "108830828338.dkr.ecr.ap-southeast-1.amazonaws.com/croptic-deep-learning:latest", # Change your container image
    secret=aws_secret,      
    force_build=True,
    )
    .pip_install(
        "modal",
        "boto3"
    )
)

@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 60,
    secrets=[aws_secret],   
)
def run_pipeline(
    user_id: str,
):
    from algorithm.pipeline import palmAnalysisPipeline

    model_path = "models/co_dino_5scale_swin_large_sawit_2024_pretrained_mopad/latest.pth"
    config_path = "models/co_dino_5scale_swin_large_sawit_2024_pretrained_mopad/co_dino_5scale_swin_large_sawit_2024_pretrained_mopad.py"

    JOB_ID = user_id
    S3_BUCKET = "croptic-user-data"   
    PREFIX = f"{JOB_ID}/"

    OUTPUT_DIR = f"/tmp/output/{JOB_ID}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    s3 = boto3.client("s3")

    # --- list .tif di {JOB_ID}/ ---
    tif_keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=PREFIX):  
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".tif") or key.lower().endswith(".tiff"):
                tif_keys.append(key)

    if not tif_keys:
        return {
            "job_id": JOB_ID,
            "runtime_sec": 0,
            "message": "No .tif files found for this user",
        }

    start = time.time()

    pipeline = palmAnalysisPipeline(
        model_path,
        config_path,
        cluster_n=7,
        min_cluster=7,
    )

    processed = []

    for key in tif_keys:
        filename = os.path.basename(key)
        stem = os.path.splitext(filename)[0]

        local_tif_path = os.path.join("/tmp", f"{JOB_ID}_{filename}")
        tif_output_dir = os.path.join(OUTPUT_DIR, stem)
        os.makedirs(tif_output_dir, exist_ok=True)

        # download tif ke local
        s3.download_file(S3_BUCKET, key, local_tif_path)  

        pipeline.run(local_tif_path, tif_output_dir)

        processed.append(
            {
                "input_key": key,
                "output_prefix": f"{JOB_ID}/{stem}/",
            }
        )

    # upload hasil ke {JOB_ID}/{nama_tif}/...
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in files:
            local = os.path.join(root, f)
            rel = os.path.relpath(local, OUTPUT_DIR)
            out_key = f"{JOB_ID}/{rel}"
            s3.upload_file(local, S3_BUCKET, out_key)

    return {
        "job_id": JOB_ID,
        "runtime_sec": time.time() - start,
        "bucket": S3_BUCKET,
        "processed": processed,
    }
