"""Create the predictor variables related with elevation for each image tile"""
from rasterio.warp import transform_bounds # type: ignore
from osgeo import gdal # type: ignore
from pathlib import Path
from io import BytesIO
from shapely.geometry import box # type: ignore
import geopandas as gpd # type: ignore
import rasterio # type: ignore
import requests
import numpy as np
import math

ROOT = Path(__file__).resolve().parent.parent
output_dir = Path(ROOT, "data/predictor_variables/dem")
images_dir = r"H:\Borini\harmoPAF\HarmoPAF_time_series"

def split_tile_by_area(gdf, target_area_h) -> list:
    """
    Split the image tiles into smaller ones. Avoid to split the tiles with an
    area less than "target_area_h".
    
    Return a GeoDataFrame of smaller bounding boxes that tile the original 
    geometry.

    :gdf: GeodataFrame.
    :target_area_m2: Desired area of each small bounding box in square meters.
    :crs: Coordinate Reference System of the gdf.
    :debugging: Export the bboxes GeoDataFrame.
    """
    # Get bounds
    minx, miny, maxx, maxy = gdf.geometry.bounds

    target_area_m2 = target_area_h * 10000
    tile_size = math.sqrt(target_area_m2)
    # Create tiles
    small_boxes = []
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            xmax = min(x + tile_size, maxx)
            ymax = min(y + tile_size, maxy)
            small_boxes.append((x, y, xmax, ymax))

            x += tile_size

        y += tile_size

    # Create GeoDataFrame
    small_boxes_shapely = [box(x, y, X, Y) for x, y, X, Y in small_boxes]
    # out_gdf = gpd.GeoDataFrame(geometry=small_boxes, crs=crs)
    # out_gdf.to_file(Path(ROOT, "data/divided_tiles_by_area.gpkg"))

    return small_boxes, small_boxes_shapely

def create_divided_tile_gpd(boxes: list, tile_names: list):
    """
    Save each subtile with the data from the parent one.

    :boxes: contains all the bounding boxes in Shapely Polygon format.
    :tile_names: Index for each tile
    """
    divided_tile = {"name": [], "subtile_fid": [], 'geometry': []}

    for i_bboxes, tile_name in zip(boxes, tile_names):
        for i, bbox in enumerate(i_bboxes):
            divided_tile["name"].append(tile_name)
            divided_tile["subtile_fid"].append(i+1)
            divided_tile["geometry"].append(bbox)

    return divided_tile

def download_dem(bounds: list, crs):
    """
    Download 5m DEM from WCS PNOA servers

    :bounds: Coordinates for xmin, ymin, xmax, ymax bounds.
    :crs: Coordinate Reference System for the bounds.
    """
    # Reproject image BBOX to the one required by the WCS
    xmin, ymin, xmax, ymax = bounds
    bounds_25830 = transform_bounds(crs,
        rasterio.crs.CRS.from_epsg(25830),
        left=xmin,
        bottom=ymin,
        right=xmax,
        top=ymax)
    # Extract each coordinate to better handling
    xmin, ymin, xmax, ymax = bounds_25830 
    # Build WCS request URL
    wcs_url = (
        "https://servicios.idee.es/wcs-inspire/mdt?"
        "service=WCS"
        "&request=GetCoverage"
        "&version=2.0.1"
        "&coverageId=Elevacion25830_25"
        f"&subset=long({xmin},{xmax})"
        f"&subset=lat({ymin},{ymax})"
        "&format=image/tiff"
    )

    # Send request to WCS
    response = requests.get(wcs_url)

    # Check response status
    if response.status_code == 200:
        print(f"Elevation data request successful.")

        # Open TIFF image from memory
        with rasterio.open(BytesIO(response.content)) as dataset:
            mdt = dataset.read(1)  # Read first band (Elevation data)
            mdt_meta = dataset.meta
            mdt_bbox = dataset.bounds

            return mdt, mdt_meta, mdt_bbox

    else:
        print(f"Error in WCS request: {response.status_code}, {response.text}")
        return False, False, False

def compute_dem_products(mdt_arr, mdt_meta, suffix, out_dir):
    """
    Save DEM locally and create slope and aspect products.
    
    The suffix parameter is used to include several DEM and derivated products from one tile which has been splitted because of its size.

    :mdt_arr: Numpy array with elevation data from download_dem()
    :mdt_meta: Metadata rasterio object from download_dem()
    :suffix: Include a number identifying the tile of the whole DEM file.
    :out_dir: Output directory to save the data.
    """
    gdal.DontUseExceptions()

    mdt_temp = Path(out_dir, f"mdt{suffix}.tif")
    with rasterio.open(mdt_temp, "w", **mdt_meta) as dst:
        dst.write(mdt_arr, 1)

    # Pendientes
    # Definir el nombre del archivo (guardado en el directorio temp)
    slope_tmp = Path(out_dir, f"slope{suffix}.tif")
    if not slope_tmp.exists():
        # Calcular las pendientes
        slope = gdal.DEMProcessing(
            slope_tmp, mdt_temp, "slope", computeEdges=True)
        # Devolver el resultado en un array numpy
        slope_arr = np.array(slope.GetRasterBand(1).ReadAsArray())
    else:
        with rasterio.open(slope_tmp) as src:
            slope_arr = src.read(1)

    # Orientaciones
    # Definir el nombre del archivo (guardado en el directorio temp)
    aspect_tmp = Path(out_dir, f"aspect{suffix}.tif")
    # Calcular las orientaciones
    # Importante: el parametro zeroForFlat debe estar desactivado. De lo 
    # contrario la IL sale al reves
    if not aspect_tmp.exists():
        aspect = gdal.DEMProcessing(aspect_tmp, mdt_temp, "aspect",
                                computeEdges=False, zeroForFlat=False)
        # Devolver el resultado en un array numpy
        aspect_arr = np.array(aspect.GetRasterBand(1).ReadAsArray())
    
    else:
        with rasterio.open(aspect_tmp) as src:
            aspect_arr = src.read(1)

    # Hillshade o sombras
    # Definir el nombre del archivo (guardado en el directorio temp)
    hill_tmp = Path(out_dir, f"hillshade{suffix}.tif")
    if not hill_tmp.exists():
        # Calcular "Shadow"
        hillshade = gdal.DEMProcessing(
            hill_tmp, mdt_temp, "hillshade", azimuth=180, altitude=45)

# Load the image tiles layer
layer_path = Path(images_dir, "tiles_perimeters", "tiles_perimeters.gpkg")
tiles = gpd.read_file(Path(layer_path))

# Create a gpd dividing each tile in sub tiles. Each sub tile will have the
# attributes from the parent one.
divided_tiles_boxes = []

for i, tile in tiles.iterrows():
    print(f"Computing tile {tile["name"]}")
    # Split each tile into boxes of 50,000 hectares
    bboxes, bboxes_shapely = split_tile_by_area(tile, 50000)
    divided_tiles_boxes.append(bboxes_shapely)
    
    # Construct the output dir
    i_out_dir = Path(output_dir, tile["name"])
    # if i_out_dir.exists():
    #     continue
    # i_out_dir.mkdir(parents=True, exist_ok=True)
    # Include a suffix to identify the splitted bboxes
    suffix = 1
    for bound in bboxes:
        print(f"Tile {str(suffix)}")
        mdt_arr, mdt_meta, mdt_bbox = download_dem(bound, tiles.crs)
        compute_dem_products(mdt_arr, mdt_meta, f"_{str(suffix)}", i_out_dir)
        print("Derivated products have been computed correctly.")
        suffix += 1

# Export the divided tiles
divided_tiles_dict = create_divided_tile_gpd(
    divided_tiles_boxes, tiles["name"].to_list())
divided_tiles_gpd = gpd.GeoDataFrame(divided_tiles_dict, crs=tiles.crs)
# Include the original tiles object data into the subtiles
divided_tiles_gpd = divided_tiles_gpd.join(
    tiles.set_index("name"), on="name", rsuffix="_dupplicated")
divided_tiles_gpd.drop(
    columns=["geometry_dupplicated"]).to_file(
        Path(ROOT, "data/divided_tiles_by_area.gpkg"))
