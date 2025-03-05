import ee
import geemap
import time
import os
import threading
from datetime import datetime, timedelta
import concurrent.futures
import geopandas as gpd

# 1. create conda 
# conda create -n gee_env python=3.10 -y
# conda activate gee_env

# 2. Install Earth Engine Python API:
# pip install earthengine-api geemap
# 3. Authenticate:
# earthengine authenticate
# 4. To confirm everything is set up, try:
# earthengine ls

# 5. Go to Google Cloud Console, create a New Project
# 6. earthengine authenticate --project=your_project_id
# Initialize Google Earth Engine (replace with your project ID)

# require input: shape file of one enitre AOI
# folder of output
# out put formattion: year-month-day / <sentinel2 file name on GEE>_<tile ID>_<long>_<la>.tif
# this script only stack 10m 3 bands
# run script: under conda env, time python3 sentinel_download_parallel_gee.py

try:
    ee.Initialize(project='') # Update with your project ID
    print("Earth Engine initialized successfully.")
except Exception as e:
    print("Error initializing Earth Engine:", e)
    raise

# === STEP 1: Read shapefile, reproject to EPSG:4326 ===
shapefile_path = "CA/CA.shp"  # Update with your shapefile path
try:
    gdf = gpd.read_file(shapefile_path)
    print("Shapefile read successfully.")
except Exception as e:
    print("Error reading shapefile:", e)
    raise

gdf = gdf.to_crs(epsg=4326)
print("Shapefile reprojected to EPSG:4326.")

# === STEP 2: Create fishnet grid for the union of features ===
aoi_shapely = gdf.unary_union
print("Created union of shapefile geometries.")

# Convert union to EE geometry.
if aoi_shapely.geom_type == 'Polygon':
    full_aoi = ee.Geometry.Polygon(list(aoi_shapely.exterior.coords))
    print("AOI is a Polygon.")
elif aoi_shapely.geom_type == 'MultiPolygon':
    # For MultiPolygon, use the exterior of the first polygon. TODO: need to update later
    full_aoi = ee.Geometry.Polygon(list(list(aoi_shapely.geoms)[0].exterior.coords))
    print("AOI is a MultiPolygon - using the first polygon's exterior.")
else:
    raise ValueError("Unsupported geometry type from shapefile.")

# Generate fishnet grid
tile_size = 0.1  # degrees; for cutting to tiles can fit GEE limitation, adjust as needed (approx. 11 km)
try:
    tiles = geemap.fishnet(full_aoi, h_interval=tile_size, v_interval=tile_size)
    print("Fishnet grid generated successfully.")
except Exception as e:
    print("Error generating fishnet grid:", e)
    raise

# Convert tiles to a native Python list for easier handling.
all_tiles = tiles.toList(tiles.size())
all_tiles_py = all_tiles.getInfo()  # Now a native Python list of dicts
num_tiles = len(all_tiles_py)
print(f"Total number of tiles generated: {num_tiles}")

# === STEP 3: Define time intervals for downloading ===
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 2, 1)
interval_days = 1  # day intervals 
delta = timedelta(days=interval_days)
current_date = start_date

# Define the tile processing function.
def process_tile(i, current_date, next_date, date_str):
    # show which thread is handling the tile.
    print(f"Tile {i} starting on thread {threading.current_thread().name}")
    
    try:
        # Retrieve the i-th tile from the Python list.
        tile_feature = all_tiles_py[i]
        tile = ee.Geometry(tile_feature['geometry'])
        print(f"Tile {i} geometry retrieved.")
    except Exception as e:
        print(f"Error retrieving tile {i}: {e}")
        return None

    try:
        # Get Sentinel-2 image collection for this tile and date interval.
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(tile)
                      .filterDate(current_date.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d'))
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        count = collection.size().getInfo()
        print(f"Tile {i} - Found {count} images.")
    except Exception as e:
        print(f"Error filtering collection for tile {i}: {e}")
        return None

    if count == 0:
        print(f"Tile {i} - No images found for {date_str}. Skipping.")
        return None

    try:
        # Create a median composite and get naming info from the first image.
        image = collection.median()
        first_image = ee.Image(collection.first())
        system_index = first_image.get('system:index').getInfo()
    except Exception as e:
        print(f"Error processing image for tile {i}: {e}")
        return None

    try:
        # Select bands for a true color composite.
        band_list = image.bandNames().getInfo()
        if 'B4' in band_list and 'B3' in band_list and 'B2' in band_list:
            image = image.select(['B4', 'B3', 'B2'])
        else:
            print(f"Tile {i} - Expected bands not found. Skipping.")
            return None
    except Exception as e:
        print(f"Error selecting bands for tile {i}: {e}")
        return None

    # Create the output folder based on the date.
    output_folder_name = "Sentinel2_CA"
    folder = os.path.join(output_folder_name, date_str)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    
    try:
        # Use tile centroid with a non-zero error margin.
        centroid = tile.centroid(maxError=1).coordinates().getInfo()  # returns [lon, lat]
        lon, lat = centroid[0], centroid[1]
        output_path = os.path.join(folder, f"{system_index}_{i}_{lon:.3f}_{lat:.3f}.tif")
    except Exception as e:
        print(f"Error computing centroid or output path for tile {i}: {e}")
        return None


    if os.path.exists(output_path):
        print(f"Tile {i}: {output_path} already exists. Skipping export.")
        return output_path

    try:
        # Export the image and measure the time.
        start_time = time.time()
        geemap.ee_export_image(
            image,
            filename=output_path,
            scale=10,         # 10 m resolution
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

# === STEP 4: Process tiles in parallel for each date interval ===
while current_date < end_date:
    next_date = current_date + delta
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"\nDownloading Sentinel-2 images for {date_str} to {next_date.strftime('%Y-%m-%d')} in tiles...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit a task for each tile.
        futures = {
            executor.submit(process_tile, i, current_date, next_date, date_str): i 
            for i in range(num_tiles)
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            # Optionally print or log the result for each tile.
    current_date = next_date

print("\nAll Sentinel-2 tiles downloaded successfully!")