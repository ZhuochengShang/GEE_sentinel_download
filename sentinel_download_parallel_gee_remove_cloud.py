import ee
import geemap
import time
import os
import threading
from datetime import datetime, timedelta
import concurrent.futures
import geopandas as gpd

# === SETUP ===
try:
    ee.Initialize(project='')  # Replace with your project ID
    print("Earth Engine initialized successfully.")
except Exception as e:
    print("---- X ---- Error initializing Earth Engine:", e)
    raise

# === STEP 1: Read Shapefile & Reproject to EPSG:4326 ===
shapefile_path = "CA/CA.shp"  
try:
    gdf = gpd.read_file(shapefile_path)
    print("Shapefile read successfully.")
except Exception as e:
    print("---- X ---- Error reading shapefile:", e)
    raise

gdf = gdf.to_crs(epsg=4326)
print("Shapefile reprojected to EPSG:4326.")

# === STEP 2: Create Fishnet Grid for AOI ===
aoi_shapely = gdf.unary_union
if aoi_shapely.geom_type == 'Polygon':
    full_aoi = ee.Geometry.Polygon(list(aoi_shapely.exterior.coords))
elif aoi_shapely.geom_type == 'MultiPolygon':
    full_aoi = ee.Geometry.Polygon(list(list(aoi_shapely.geoms)[0].exterior.coords))
else:
    raise ValueError("---- X ---- Unsupported geometry type from shapefile.")

tile_size = 0.1  
try:
    tiles = geemap.fishnet(full_aoi, h_interval=tile_size, v_interval=tile_size)
    print(f"Fishnet grid generated successfully with {tiles.size().getInfo()} tiles.")
except Exception as e:
    print("---- X ---- Error generating fishnet grid:", e)
    raise

all_tiles = tiles.toList(tiles.size()).getInfo()
num_tiles = len(all_tiles)

# === STEP 3: Define Time Intervals ===
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 2, 1)
interval_days = 1
delta = timedelta(days=interval_days)
current_date = start_date

# === FUNCTION TO APPLY SCL CLOUD MASK ===
def mask_clouds_SCL(image):
    """Applies cloud masking using only the Scene Classification Layer (SCL)."""
    scl = image.select('SCL')
    scl_mask = scl.neq(9).And(scl.neq(3))  # Remove clouds (9) & cloud shadows (3)
    return image.updateMask(scl_mask)

# === FUNCTION TO REMOVE PIXELS WITH NODATA IN ANY BAND ===
def remove_nodata_pixels(image):
    """Ensures all four bands (B4, B3, B2, B8) contain valid pixels."""
    valid_mask = (image.select('B4').mask()
                  .And(image.select('B3').mask())
                  .And(image.select('B2').mask())
                  .And(image.select('B8').mask()))
    return image.updateMask(valid_mask)

# === FUNCTION TO CHECK IF 80% OF PIXELS ARE VALID ===
def has_80_percent_valid_pixels(image, tile):
    """Ensures at least 80% of pixels in RGB (B4, B3, B2) and NIR (B8) are valid."""
    bands = ['B4', 'B3', 'B2', 'B8']
    valid_pixel_counts = []

    for band in bands:
        valid_mask = image.select(band).mask()

        valid_pixel_count = valid_mask.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=tile, scale=10, maxPixels=1e8
        ).values().get(0) or 0

        total_pixel_count = valid_mask.reduceRegion(
            reducer=ee.Reducer.count(), geometry=tile, scale=10, maxPixels=1e8
        ).values().get(0) or 1  # Avoid division by zero

        valid_percentage = ee.Algorithms.If(
            ee.Number(total_pixel_count).neq(0), 
            ee.Number(valid_pixel_count).divide(total_pixel_count).multiply(100), 
            0
        )

        valid_pixel_counts.append(valid_percentage)

    min_valid_percent = ee.Number(ee.List(valid_pixel_counts).reduce(ee.Reducer.min()))  # Ensure it's an ee.Number
    return min_valid_percent.gte(80)  # Now safely compare it with 80

# === STEP 4: Processing Function ===
def process_tile(i, current_date, next_date, date_str):
    print(f"Tile {i} starting on thread {threading.current_thread().name}")

    try:
        tile_feature = all_tiles[i]
        tile = ee.Geometry(tile_feature['geometry'])
    except Exception as e:
        print(f"---- X ---- Error retrieving tile {i}: {e}")
        return None

    try:
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(tile)
                      .filterDate(current_date.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d'))
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1.01))
                      .map(mask_clouds_SCL)
                      .map(remove_nodata_pixels))  # Ensures NoData pixels are removed

        count = collection.size().getInfo()
    except Exception as e:
        print(f"---- X ---- Error filtering collection for tile {i}: {e}")
        return None

    if count == 0:
        print(f"Tile {i} - No images found for {date_str}. Skipping.")
        return None

    try:
        collection = collection.sort('CLOUDY_PIXEL_PERCENTAGE', True)
        best_image = collection.first().select(['B4', 'B3', 'B2', 'B8'])  # **Only select RGB + NIR bands**

        # **Check if at least 80% of pixels are valid before downloading**
        if not has_80_percent_valid_pixels(best_image, tile).getInfo():
            print(f"Tile {i} - Less than 80% valid pixels. Skipping.")
            return None

        system_index = best_image.get('system:index').getInfo()
    except Exception as e:
        print(f"---- X ---- Error processing image for tile {i}: {e}")
        return None

    print(f"Tile {i} - Selected best image with at least 80% valid pixels.")

    # **EXPORT THE IMAGE**
    try:
        output_path = f"Sentinel2_CA/{date_str}/{system_index}_{i}.tif"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Downloading image for Tile {i}: {output_path}")
        geemap.ee_export_image(best_image, filename=output_path, scale=10, region=tile, file_per_band=False)
        print(f"Downloaded Tile {i}: {output_path}")
    except Exception as e:
        print(f"---- X ---- Error exporting tile {i}: {e}")
        return None

    return output_path

# === STEP 5: Run in Parallel ===
while current_date < end_date:
    next_date = current_date + delta
    date_str = current_date.strftime('%Y-%m-%d')

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_tile, i, current_date, next_date, date_str): i for i in range(num_tiles)}
        for future in concurrent.futures.as_completed(futures):
            future.result()

    current_date = next_date

print("\n All Sentinel-2 tiles processed and downloaded successfully!")




