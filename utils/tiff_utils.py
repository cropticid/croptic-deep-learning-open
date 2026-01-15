import numpy as np
import rasterio

def load_tif_image(image_path: str) -> np.ndarray:
    """
    Loads a TIFF/GeoTIFF image (single-band or multi-band) and returns a NumPy array as RGB uint8.

    Args:
        image_path (str): Path to a .tif or .tiff file.

    Returns:
        np.ndarray: An (H, W, 3) uint8 array for RGB visualization/processing.
    """
    with rasterio.open(image_path) as src:
        arr = src.read()  # (bands, height, width)
        if arr.shape[0] == 3:
            img = np.transpose(arr, (1, 2, 0))  # RGB order
        elif arr.shape[0] == 1:
            # Single band: tile to (H, W, 3)
            img = np.repeat(arr[0, :, :][..., None], 3, axis=-1)
        elif arr.shape[0] > 3:
            # Take first 3 as RGB
            img = np.transpose(arr[:3], (1, 2, 0))
        else:
            raise ValueError("TIFF image has no bands.")
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img
