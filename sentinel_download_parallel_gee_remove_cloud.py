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
    print("Error initializing Earth Engine:", e)
    raise

# === STEP 1: Read Shapefile & Reproject to EPSG:4326 ===
shapefile_path = "CA/CA.shp"  # Update with your shapefile path
try:
    gdf = gpd.read_file(shapefile_path)
    print("Shapefile read successfully.")
except Exception as e:
    print("Error reading shapefile:", e)
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
    raise ValueError("Unsupported geometry type from shapefile.")

tile_size = 0.1  # Approx. 11 km tiles
try:
    tiles = geemap.fishnet(full_aoi, h_interval=tile_size, v_interval=tile_size)
    print("Fishnet grid generated successfully.")
except Exception as e:
    print("Error generating fishnet grid:", e)
    raise

all_tiles = tiles.toList(tiles.size()).getInfo()
num_tiles = len(all_tiles)
print(f"Total number of tiles generated: {num_tiles}")

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

# === FUNCTION TO CHECK IF ALL BANDS HAVE VALID DATA ===
def check_valid_bands(img, tile):
    """Ensures all bands (B4, B3, B2, B8) have at least 80% valid (non-null) pixels."""
    bands = ['B4', 'B3', 'B2', 'B8']  # RGB + NIR
    valid_percentages = ee.List([])  # Initialize empty EE list

    for band in bands:
        valid_mask = img.select(band).mask()

        valid_pixel_count = ee.Number(valid_mask.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=tile, scale=10, maxPixels=1e8
        ).values().get(0) or 0)  # Ensure no None values

        total_pixel_count = ee.Number(valid_mask.reduceRegion(
            reducer=ee.Reducer.count(), geometry=tile, scale=10, maxPixels=1e8
        ).values().get(0) or 1)  # Avoid division by zero

        valid_percentage = ee.Algorithms.If(
            total_pixel_count.neq(0), valid_pixel_count.divide(total_pixel_count).multiply(100), 0
        )  # Handle cases where total_pixel_count is zero

        valid_percentages = valid_percentages.add(valid_percentage)

    # Compute the **minimum** valid percentage across all bands using Earth Engine
    min_valid_percent = valid_percentages.reduce(ee.Reducer.min())
    return img.set('valid_percentage', min_valid_percent)

# === STEP 4: Processing Function ===
def process_tile(i, current_date, next_date, date_str):
    print(f"Tile {i} starting on thread {threading.current_thread().name}")

    try:
        tile_feature = all_tiles[i]
        tile = ee.Geometry(tile_feature['geometry'])
    except Exception as e:
        print(f"Error retrieving tile {i}: {e}")
        return None

    try:
        # Get images with ≤ 1% cloud cover and apply SCL mask
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(tile)
                      .filterDate(current_date.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d'))
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1.01))
                      .map(mask_clouds_SCL)
                      .map(lambda img: check_valid_bands(img, tile)))  # Ensure all bands have >80% valid data

        count = collection.size().getInfo()
    except Exception as e:
        print(f"Error filtering collection for tile {i}: {e}")
        return None

    if count == 0:
        print(f"Tile {i} - No images found for {date_str}. Skipping.")
        return None

    try:
        collection = collection.sort('valid_percentage', False)  # Best images first
        best_image = collection.first()
        system_index = best_image.get('system:index').getInfo()
        valid_percentage = best_image.get('valid_percentage').getInfo()
    except Exception as e:
        print(f"Error processing image for tile {i}: {e}")
        return None

    if valid_percentage < 80:  # Reject images if any band has <80% valid pixels
        print(f"Tile {i} - One or more bands have too many empty pixels ({valid_percentage:.2f}%). Skipping.")
        return None

    print(f"Tile {i} - Selected best image with {valid_percentage:.2f}% valid pixels.")

    # Define output file
    output_folder_name = "Sentinel2_CA"
    folder = os.path.join(output_folder_name, date_str)
    if not os.path.exists(folder):
        os.makedirs(folder)

    centroid = tile.centroid(maxError=1).coordinates().getInfo()
    lon, lat = centroid[0], centroid[1]
    output_path = os.path.join(folder, f"{system_index}_{i}_{lon:.3f}_{lat:.3f}.tif")

    if os.path.exists(output_path):
        print(f"Tile {i}: {output_path} already exists. Skipping export.")
        return output_path

    # **EXPORT THE IMAGE**
    try:
        print(f"Downloading image for Tile {i}: {output_path}")
        start_time = time.time()
        geemap.ee_export_image(
            best_image,
            filename=output_path,
            scale=10,  # 10m resolution
            region=tile,
            file_per_band=False
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Downloaded Tile {i}: {output_path} in {elapsed_time:.2f} seconds")
    except Exception as e:
        print(f"Error exporting tile {i}: {e}")
        return None

    return output_path

# === STEP 5: Run in Parallel ===
while current_date < end_date:
    next_date = current_date + delta
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"\nProcessing Sentinel-2 images for {date_str} in tiles...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_tile, i, current_date, next_date, date_str): i for i in range(num_tiles)}
        for future in concurrent.futures.as_completed(futures):
            future.result()

    current_date = next_date

print("\nAll Sentinel-2 tiles processed and downloaded successfully!")




