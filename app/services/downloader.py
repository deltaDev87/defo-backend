import json
import math
import shutil
from pathlib import Path

from pystac_client import Client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


BASE_PATH = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_PATH / "config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

INPUT_ROOT = BASE_PATH / config["input_folder"]
INPUT_ROOT.mkdir(parents=True, exist_ok=True)

FOLDER_1 = INPUT_ROOT / config["start_folder"]
FOLDER_2 = INPUT_ROOT / config["end_folder"]


def clean_previous_inputs():
    for folder in [FOLDER_1, FOLDER_2]:
        if folder.exists():
            shutil.rmtree(folder)


def get_landsat_item(date, bbox):
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    datetime_range = f"{date}-01/{date}-28"

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": 60}},
        limit=20
    )

    items = list(search.items())

    if not items:
        raise Exception(f"No scenes found for {datetime_range}")

    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))

    return items[0]


def crop_and_save(url, output_path, bbox):
    with rasterio.open(url) as src:

        transformed_bbox = transform_bounds(
            "EPSG:4326",
            src.crs,
            *bbox
        )

        raster_bounds = src.bounds

        left = max(transformed_bbox[0], raster_bounds.left)
        bottom = max(transformed_bbox[1], raster_bounds.bottom)
        right = min(transformed_bbox[2], raster_bounds.right)
        top = min(transformed_bbox[3], raster_bounds.top)

        if left >= right or bottom >= top:
            raise Exception("BBOX does not intersect raster footprint.")

        window = from_bounds(left, bottom, right, top, src.transform)

        data = src.read(1, window=window)

        if data.size == 0:
            raise Exception("Crop returned empty data.")

        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update({
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)


def download_bands(item, folder, bbox):
    signed_item = planetary_computer.sign(item)

    blue_url = signed_item.assets["blue"].href
    red_url = signed_item.assets["red"].href
    nir_url = signed_item.assets["nir08"].href

    crop_and_save(blue_url, folder / "band2.TIF", bbox)
    crop_and_save(red_url, folder / "band4.TIF", bbox)
    crop_and_save(nir_url, folder / "band5.TIF", bbox)


def download_data(latitude, longitude, start_date, end_date, box_size_km ):
    if box_size_km > 90:
        raise ValueError("Box size must be less than or equal to 90 km.")

    delta_lat = box_size_km / 111
    delta_lon = box_size_km / (111 * math.cos(math.radians(latitude)))

    bbox = [
        longitude - delta_lon / 2,
        latitude - delta_lat / 2,
        longitude + delta_lon / 2,
        latitude + delta_lat / 2
    ]

    clean_previous_inputs()

    item1 = get_landsat_item(start_date, bbox)
    download_bands(item1, FOLDER_1, bbox)

    item2 = get_landsat_item(end_date, bbox)
    download_bands(item2, FOLDER_2, bbox)

    return {
        "status": "download_complete",
        "bbox": bbox
    }
