"""
Automatic extraction of soil with sparse vegetation points.
"""
from rasterio.io import MemoryFile # type: ignore
from scipy.spatial import KDTree # type: ignore
from shapely import Point # type: ignore
from pathlib import Path

import geopandas as gpd # type: ignore
import numpy as np # type: ignore
import rasterio # type: ignore

from utils import Tile, Rasters
from utils_il import IL

ROOT = Path(__file__).resolve().parent.parent
# Output path to store extracted points
out_path = Path(ROOT, r"data\labels\ground_points_db.gpkg")

# Load divided tiles (the ones created with "create_dem.py")
tiles_path = Path(ROOT, "data/divided_tiles_by_area.gpkg")
tiles = gpd.read_file(tiles_path)
# Borini images path
images_path = r"H:\Borini\harmoPAF\HarmoPAF_time_series"

# Images to compute IL image
# It is used to remove low NDVI values produced by shadows (not soil)
aspect_path = Path(ROOT, "data/predictor_variables/dem/aspect.vrt")
slope_path = Path(ROOT, "data/predictor_variables/dem/slope.vrt")

# SIOSE target codes (the ones related to bare soils)
valid_codiige = {
    "roquedo": [217, 214, 199],
    "temporalmente_desarbolado_por_incendios": [60, 80, 60], # No existe en Aragon para 2005 a 2014
    "suelo_desnudo": [210, 242, 194]
}

siose_years = [2005, 2009, 2011, 2014]

def filter_by_distance(df: gpd.GeoDataFrame, distance = 200):
    """
    Using query_ball_point to find all points without certain distance,
    and keeping only one of them.
    """
    # Extract coordinates
    points = np.vstack(df.geometry.apply(lambda geom: (geom.x, geom.y)))
    
    # Step 1: Build the KDTree
    tree = KDTree(points)

    # Find clusters: Get all neighbors within 150m
    neighbors_list = tree.query_ball_point(points, r=distance)

    # Step 2: Efficient Filtering (Greedy selection)
    selected = np.zeros(len(points), dtype=bool)  # Boolean mask to track kept points
    visited = np.zeros(len(points), dtype=bool)  # Tracks points that are processed

    for i in range(len(points)):
        if visited[i]: 
            continue  # Skip points already removed
        
        selected[i] = True  # Keep this point
        visited[neighbors_list[i]] = True  # Mark all neighbors as visited

    # Return the valid points (the ones without nearest neighbor)
    return df.loc[selected, :].copy()

def extract_ndvi_points(ndvi: np.array, img_meta):
    """
    Extract the NDVI values from a rasterio array. Get the centroid of each
    pixel and save the NDVI data.
    """
    # Define the NDVI threshold (select only bare soil)
    mask = (ndvi >= 0.08) & (ndvi <= 0.15)

    # Get row and column indices of the selected pixels
    rows, cols = np.where(mask)

    # Convert row, col indices to x, y coordinates
    x_list, y_list = rasterio.transform.xy(img_meta["transform"], rows, cols)

    # Extract the NDVI values of selected pixels
    ndvi_values = ndvi[rows, cols]

    # Create a GeoDataFrame with the extracted points
    gdf = gpd.GeoDataFrame(
        {"NDVI": ndvi_values},
        geometry=[Point(x, y) for x, y in zip(x_list, y_list)],
        crs=img_meta["crs"]
    )
    return gdf

for i, tile in tiles.iterrows():

    print(f"Process tile {tile["name"]}, subtile {tile["subtile_fid"]}")
    tile_obj = Tile(Path(images_path, tile["name"]))
    tile_years = tile_obj.get_years()

    for siose_year in siose_years:
        # Get next year if the siose's year is not inside tile year
        if siose_year not in tile_years:
            continue

        # Generate a mean composite with all the year
        # Select summer interval in order to reduce shadows
        filter_dates = [f'{siose_year}-{s}' for s in ["06-01", "07-31"]]
        img_props = tile_obj.filter_date(filter_dates[0], filter_dates[1])
        # Load images inside the tile
        tile_bbox = tile.geometry.bounds
        images = Rasters(img_props, tile_obj.imgs_dir, tile_bbox, tiles.crs)
        images.compute_mean()
        images.compute_ndvi()

        # Extract only NDVI values related to bare soil
        # The last band of the composite is the NDVI
        ndvi_pnts = extract_ndvi_points(
            images.composite[-1, :, :], images.composite_meta)
        
        # Avoid processing empty geodataframes
        if ndvi_pnts.shape[0] == 0:
            continue

        # Filter points by distance avoiding spatial autocorrelation
        ndvi_pnts = filter_by_distance(ndvi_pnts)
        ndvi_pnts.loc[:, "YEAR"] = siose_year
        
        # Obtain the IL array
        il = IL(img_props, images.bounds, images.composite_meta["crs"])
        il_array = il.compute(aspect_path, slope_path)
        il_meta = images.composite_meta.copy()
        il_meta.update({
            "count": 1,
            "dtype": il_array.dtype
        })
        # Create an in-memory raster. This allows to write a # NumPy array and
        # metadata into a virtual file and use all Rasterio features like
        #  sample() as if it were a real file.
        with MemoryFile() as memfile:
            with memfile.open(**il_meta) as dst:
                # Add one band at the beginning to match rasterio format
                dst.write(il_array[None, :, :])
                
                # Extract IL values intersecting the NDVI points,
                # reproject them to the IL image CRS first.
                pnts_repr = ndvi_pnts.to_crs(dst.meta["crs"]).geometry
                # Transform coordinates into a list
                # coords = [(x, y) for x, y in zip(pnts_repr.x, pnts_repr.y)]
                # More efficient than the above line
                coords = np.column_stack((pnts_repr.x.values, pnts_repr.y.values))
                # Extract IL values
                il_values = list(dst.sample(coords, indexes=1))
        
        ndvi_pnts.loc[:, "IL"] = np.array(il_values)
        ndvi_pnts = ndvi_pnts.query("IL > 0.7")

        if ndvi_pnts.shape[0] == 0:
            continue

        # Get SIOSE image array
        siose_path = Path(ROOT, "data/siose", tile["name"],
                          f"siose{siose_year}.tif")
        

        with rasterio.open(siose_path) as src:

            # Extract SIOSE values in each ground point
            # Reproject them to the SIOSE image CRS first.
            pnts_repr = ndvi_pnts.to_crs(src.meta["crs"]).geometry
            # Transform coordinates into a list
            # coords = [(x, y) for x, y in zip(pnts_repr.x, pnts_repr.y)]
            # More efficient than the above line
            coords = np.column_stack((pnts_repr.x.values, pnts_repr.y.values))
            # Extract siose values
            siose_values = list(src.sample(coords))

        # The SIOSE array has three bands: RGB
        siose_bands = ["R", "G", "B"]
        ndvi_pnts.loc[:, siose_bands] = np.array(siose_values)
        ndvi_pnts.to_file(Path(ROOT, "test.gpkg"))
        # Filter points by SIOSE values
        for siose_code, [r, g, b] in valid_codiige.items():
            
            siose_query = f"R == {r} &  G == {g} & B == {b}"
            ground_pnts = ndvi_pnts.query(siose_query).copy()

            # Avoiding save empty dataframes
            if ground_pnts.shape[0] == 0:
                continue
            
            # Remove RGB columns by the its real value
            ground_pnts.drop(columns=["R", "G", "B"], inplace=True)
            ground_pnts["siose_codiige"] = siose_code

            # Output filtered points
            if Path(out_path).exists():
                ground_pnts.to_file(out_path, mode='a')
            else:
                ground_pnts.to_file(out_path)
