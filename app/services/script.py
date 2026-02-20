from timeit import main
import rasterio
import numpy as np
import json
from pathlib import Path
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import time
from rasterio.warp import reproject, Resampling
import shutil


BASE_PATH = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_PATH / "config.json"

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)


L = config["L"]
NDVI_THRESHOLD = config["ndvi_threshold"]
SAVI_THRESHOLD = config["savi_threshold"]
DEFORESTATION_ALERT_THRESHOLD = config["deforestation_alert_threshold"]
CLOUD_SHADOW_THRESHOLD = config["cloud_shadow_threshold"]

start_folder = Path(config["start_folder"])
end_folder = Path(config["end_folder"])
input_folder = Path(config["input_folder"])
output_folder = Path(config["output_folder"])

META_NODATA = -9999.0


def apply_cloud_shadow_mask(red, nir):
    brightness = (red + nir) / 2
    mask = brightness < CLOUD_SHADOW_THRESHOLD
    red = np.where(mask, np.nan, red)
    nir = np.where(mask, np.nan, nir)
    return red, nir


def dark_object_subtraction(band):
    dark_value = np.nanpercentile(band, 1)
    corrected = band - dark_value
    corrected[corrected < 0] = 0
    return corrected


def save_as_png(data, output_png_path):
    clipped = np.clip(data, -1, 1)
    normalized = ((clipped + 1) / 2 * 255)
    normalized = np.where(np.isfinite(normalized), normalized, 0)
    normalized = normalized.astype(np.uint8)
    cv2.imwrite(str(output_png_path), normalized)
    logging.info(f"Saved PNG: {output_png_path}")


def compute_indices(red_path, nir_path, ndvi_out, savi_out, ndvi_png, savi_png):
    if not red_path.exists():
        raise FileNotFoundError(red_path)
    if not nir_path.exists():
        raise FileNotFoundError(nir_path)

    with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:

        red = red_src.read(1).astype(np.float32)
        nir = nir_src.read(1).astype(np.float32)

        if red_src.nodata is not None:
            red[red == red_src.nodata] = np.nan
        if nir_src.nodata is not None:
            nir[nir == nir_src.nodata] = np.nan

        red = dark_object_subtraction(red)
        nir = dark_object_subtraction(nir)

        red, nir = apply_cloud_shadow_mask(red, nir)

        np.seterr(divide='ignore', invalid='ignore')

        ndvi = (nir - red) / (nir + red)
        savi = ((nir - red) / (nir + red + L)) * (1 + L)

        meta = red_src.meta.copy()
        meta.update(dtype=rasterio.float32, count=1, nodata=META_NODATA)

        ndvi_write = np.where(np.isfinite(ndvi), ndvi, META_NODATA)
        savi_write = np.where(np.isfinite(savi), savi, META_NODATA)

        with rasterio.open(ndvi_out, 'w', **meta) as dst:
            dst.write(ndvi_write.astype(np.float32), 1)

        with rasterio.open(savi_out, 'w', **meta) as dst:
            dst.write(savi_write.astype(np.float32), 1)

        save_as_png(ndvi, ndvi_png)
        save_as_png(savi, savi_png)

    logging.info(f"Computed NDVI/SAVI for {red_path.name}")


def compare_indices(old_path, new_path, output_path):

    with rasterio.open(old_path) as old_src, rasterio.open(new_path) as new_src:

        old_data = old_src.read(1).astype(np.float32)
        old_data[old_data == old_src.nodata] = np.nan

        new_resampled = np.full_like(old_data, np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(new_src, 1),
            destination=new_resampled,
            src_transform=new_src.transform,
            src_crs=new_src.crs,
            dst_transform=old_src.transform,
            dst_crs=old_src.crs,
            resampling=Resampling.bilinear
        )

        new_resampled[new_resampled == new_src.nodata] = np.nan

        change = new_resampled - old_data
        change_write = np.where(np.isfinite(change), change, META_NODATA)

        meta = old_src.meta.copy()
        meta.update(dtype=rasterio.float32, count=1, nodata=META_NODATA)

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(change_write.astype(np.float32), 1)

    logging.info(f"Created aligned change raster: {output_path.name}")


def detect_deforestation(ndvi_change_path, savi_change_path):
    with rasterio.open(ndvi_change_path) as ndvi_src, \
         rasterio.open(savi_change_path) as savi_src:

        ndvi_change = ndvi_src.read(1).astype(np.float32)
        savi_change = savi_src.read(1).astype(np.float32)

        ndvi_change[ndvi_change == META_NODATA] = np.nan
        savi_change[savi_change == META_NODATA] = np.nan

        valid_mask = np.isfinite(ndvi_change) & np.isfinite(savi_change)
        total_pixels = np.sum(valid_mask)

        if total_pixels == 0:
            return {
                "deforestation_percentage": 0,
                "status": "No valid pixels found."
            }

        mask = (ndvi_change < -abs(NDVI_THRESHOLD)) & \
               (savi_change < -abs(SAVI_THRESHOLD))

        deforested_pixels = np.sum(mask & valid_mask)
        percentage = round((deforested_pixels / total_pixels) * 100, 2)

        result = {
            "deforestation_percentage": percentage,
            "status": (
                "Significant deforestation detected!"
                if percentage > DEFORESTATION_ALERT_THRESHOLD
                else "No significant deforestation detected."
            )
        }

        return result


#cleans prev output files
def clean_previous_outputs(base_path):
    output_root = base_path / output_folder

    if output_root.exists():
        logging.info("Cleaning previous output files...")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)



# main fxn

def run_pipeline():

    base_path = BASE_PATH

    # Clean previous
    clean_previous_outputs(base_path)

    #path for all files
    paths = {
        "red_old": base_path / input_folder / start_folder / "band4.TIF",
        "nir_old": base_path / input_folder / start_folder / "band5.TIF",
        "red_new": base_path / input_folder / end_folder / "band4.TIF",
        "nir_new": base_path / input_folder / end_folder / "band5.TIF",

        "ndvi_old": base_path / output_folder / start_folder / "ndvi.TIF",
        "savi_old": base_path / output_folder / start_folder / "savi.TIF",
        "ndvi_new": base_path / output_folder / end_folder / "ndvi.TIF",
        "savi_new": base_path / output_folder / end_folder / "savi.TIF",
    }


    paths["ndvi_old"].parent.mkdir(parents=True, exist_ok=True)
    paths["ndvi_new"].parent.mkdir(parents=True, exist_ok=True)


    compute_indices(
        paths["red_old"],
        paths["nir_old"],
        paths["ndvi_old"],
        paths["savi_old"],
        paths["ndvi_old"].with_suffix(".png"),
        paths["savi_old"].with_suffix(".png"),
    )

    compute_indices(
        paths["red_new"],
        paths["nir_new"],
        paths["ndvi_new"],
        paths["savi_new"],
        paths["ndvi_new"].with_suffix(".png"),
        paths["savi_new"].with_suffix(".png"),
    )


    ndvi_change = base_path / output_folder / "ndvi_change.TIF"
    savi_change = base_path / output_folder / "savi_change.TIF"

    compare_indices(paths["ndvi_old"], paths["ndvi_new"], ndvi_change)
    compare_indices(paths["savi_old"], paths["savi_new"], savi_change)

    result = detect_deforestation(ndvi_change, savi_change)

    return result

##script entry point
