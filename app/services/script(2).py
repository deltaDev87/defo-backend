import rasterio
import numpy as np
import time
import json
import sys
from pathlib import Path  
import cv2
from concurrent.futures import ThreadPoolExecutor

start_folder = Path(sys.argv[1])
end_folder = Path(sys.argv[2])

def save_as_png(data, output_png_path):
    min_val, max_val = np.nanmin(data), np.nanmax(data)
    normalized = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    cv2.imwrite(str(output_png_path), normalized)

def compute_indices(red_band_path, nir_band_path, output_ndvi_path, output_savi_path, ndvi_png_path, savi_png_path, L=0.5):
    with rasterio.open(red_band_path, mmap=True, num_threads="all_cpus") as red_src, \
         rasterio.open(nir_band_path, mmap=True, num_threads="all_cpus") as nir_src:
        red = red_src.read(1).astype(np.float32)
        nir = nir_src.read(1).astype(np.float32)

        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (nir - red) / (nir + red)
        savi = ((nir - red) / (nir + red + L)) * (1 + L)

        meta = red_src.meta
        meta.update(dtype=rasterio.float32, count=1)

        with rasterio.open(output_ndvi_path, 'w', **meta, num_threads="all_cpus") as ndvi_dst:
            ndvi_dst.write(ndvi, 1)
        with rasterio.open(output_savi_path, 'w', **meta, num_threads="all_cpus") as savi_dst:
            savi_dst.write(savi, 1)
    
    save_as_png(ndvi, ndvi_png_path)
    save_as_png(savi, savi_png_path)

def compare_indices(old_path, new_path, output_change_path):
    with rasterio.open(old_path, mmap=True, num_threads="all_cpus") as old_src, \
         rasterio.open(new_path, mmap=True, num_threads="all_cpus") as new_src:
        old_data, new_data = old_src.read(1).astype(np.float32), new_src.read(1).astype(np.float32)
        min_rows, min_cols = min(old_data.shape[0], new_data.shape[0]), min(old_data.shape[1], new_data.shape[1])
        change = new_data[:min_rows, :min_cols] - old_data[:min_rows, :min_cols]
        meta = old_src.meta
        meta.update(dtype=rasterio.float32, count=1, height=min_rows, width=min_cols)
        with rasterio.open(output_change_path, 'w', **meta, num_threads="all_cpus") as dst:
            dst.write(change, 1)

def detect_deforestation(ndvi_change_path, savi_change_path, ndvi_threshold=-0.2, savi_threshold=-0.2):
    with rasterio.open(ndvi_change_path, mmap=True, num_threads="all_cpus") as ndvi_src, \
         rasterio.open(savi_change_path, mmap=True, num_threads="all_cpus") as savi_src:
        ndvi_change, savi_change = ndvi_src.read(1), savi_src.read(1)
        deforested_pixels = np.sum((ndvi_change < ndvi_threshold) & (savi_change < savi_threshold))
        deforestation_percentage = round((deforested_pixels / ndvi_change.size) * 100, 2)
        result = {
            "deforestation_percentage": deforestation_percentage,
            "status": "🚨 Significant deforestation detected!" if deforestation_percentage > 5 else "✅ No significant deforestation detected."
        }
    print(json.dumps(result))
    sys.stdout.flush()

def parallel_processes():
    base_path = Path(__file__).resolve().parent
    paths = {
        "red_band_old": base_path / "NDVI B4 B5" / start_folder / "band4.TIF",
        "nir_band_old": base_path / "NDVI B4 B5" / start_folder / "band5.TIF",
        "red_band_new": base_path / "NDVI B4 B5" / end_folder / "band4.TIF",
        "nir_band_new": base_path / "NDVI B4 B5" / end_folder / "band5.TIF",
        "ndvi_old": base_path / "temp_results/ndvis/ndvi-old.tif",
        "ndvi_new": base_path / "temp_results/ndvis/ndvi-new.tif",
        "ndvi_change": base_path / "temp_results/ndvis/ndvi_change.tif",
        "savi_old": base_path / "temp_results/savis/savi-old.tif",
        "savi_new": base_path / "temp_results/savis/savi-new.tif",
        "savi_change": base_path / "temp_results/savis/savi_change.tif",
    }

    # Ensure directories exist
    Path(paths["ndvi_old"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["savi_old"]).parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor() as executor:
        executor.submit(compute_indices, paths["red_band_old"], paths["nir_band_old"], paths["ndvi_old"], paths["savi_old"], paths["ndvi_old"].with_suffix(".png"), paths["savi_old"].with_suffix(".png"))
        executor.submit(compute_indices, paths["red_band_new"], paths["nir_band_new"], paths["ndvi_new"], paths["savi_new"], paths["ndvi_new"].with_suffix(".png"), paths["savi_new"].with_suffix(".png"))

    compare_indices(paths["ndvi_old"], paths["ndvi_new"], paths["ndvi_change"])
    compare_indices(paths["savi_old"], paths["savi_new"], paths["savi_change"])
    detect_deforestation(paths["ndvi_change"], paths["savi_change"])

if __name__ == "__main__":
    parallel_processes()