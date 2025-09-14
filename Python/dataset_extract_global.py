"""
Merge datasets by Landsat tiles and include info from global predictors

The code datastet_extract_landsat.py must have been executed.
"""
from pathlib import Path
import rasterio
import pandas as pd
import geopandas as gpd

ROOT = Path(Path(__file__).parents[1])

# Folder storing all the tile Landsat data
extracted_tile_folder = Path(ROOT, "results/tile_extracted_values")

# Read geopackages
gpkgs = [gpd.read_file(f) for f in extracted_tile_folder.glob("*.gpkg")]
# Cast all points into a only one CRS
gpkgs = [gdf.to_crs(4326) for gdf in gpkgs]

dataset = pd.concat(gpkgs, ignore_index=True)
original_columns = dataset.columns

# Counting null values by row, keep the row with less null values
dataset["_nulls"] = dataset.isna().sum(axis=1)

# Get exact coordinates
dataset["_geom"] = dataset.geometry.apply(lambda geom: geom.wkt)

# Remove duplicate rows
# Sort by number of na values by column to keep the one with less na values (the first)
dataset_sorted = dataset.sort_values("_nulls")

# Define columns to search duplicate values
subset_cols = ["YEAR", "source", "_geom"]

# Drop duplicates and keep the first one (the one with less null values per row)
unique_dat = dataset_sorted.drop_duplicates(subset=subset_cols, keep="first").copy()
unique_dat = unique_dat[original_columns]

def sample_image_into_points(
    gdf: gpd.GeoDataFrame,
    variable_name,
    img_path
) -> gpd.GeoDataFrame:
    """
    For each image, sample it into points of the GeoDataFrame. 
    Results are stored in a single column.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must have point geometries, a valid CRS, and a column 'YEAR'.
    variable_name : str
        The name of the variable to include as a new column.
    img_path : Path
        Image path.

    Returns
    -------
    GeoDataFrame
        Same GeoDataFrame with a new column containing sampled values.
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a CRS defined.")

    with rasterio.open(img_path) as src:

        gdf_prj = gdf.to_crs(src.crs).copy()
    
        # Get point coordinates
        coords = [(geom.x, geom.y) for geom in gdf_prj.geometry]

        # Extract values
        values = list(src.sample(coords, indexes=1))
    
    # Create output column if missing
    if variable_name not in gdf.columns:
        gdf[variable_name] = [v[0] for v in values]
    else:
        raise ValueError("The new column cannot be inside the dataframe.")
    
    return gdf

# Extract dem derivates predictor variables
for p in Path(ROOT, "data/predictor_variables/dem").glob("*.vrt"):
    variable = p.stem
    print(f"Extract {variable}")
    unique_dat = sample_image_into_points(unique_dat, variable, p)

# Extract ACIBASI
acibasi = Path(ROOT, "data/predictor_variables/geologia/ACIBASI.tif")
print("Extract acibasi")
unique_dat = sample_image_into_points(unique_dat, "acibasi", acibasi)

out_path = Path(ROOT, "results/dataset_with_values.gpkg")
unique_dat.to_file(out_path, index=False)
