import os
import subprocess
import sys

def run(cmd):
    print(f"\n🚀 {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("=== Croptic Local Installation ===")

    # 1️⃣ Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required.")
        sys.exit(1)

    # 2️⃣ Check Torch
    try:
        import torch
        print(f"✅ Torch version: {torch.__version__}")
    except ImportError:
        print("⚠️ Torch not installed. Please install manually, e.g.:")
        print("   pip install torch==1.11.0+cu113 torchvision==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html")
        sys.exit(1)

    # 3️⃣ Install requirements.txt
    if os.path.exists("requirements.txt"):
        run("pip install --no-cache-dir -r requirements.txt")
    else:
        print("⚠️ requirements.txt not found!")

    # 4️⃣ Install local packages (editable mode)
    local_pkg = "packages/Co-DETR"
    if os.path.exists(local_pkg):
        run(f"pip install --no-build-isolation -e {local_pkg}")
    else:
        print(f"⚠️ Local package not found at {local_pkg}")

    print("\n✅ Installation complete!")

if __name__ == "__main__":
    main()
