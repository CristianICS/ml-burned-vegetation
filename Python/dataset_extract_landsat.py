"""
Extract Landsat composite values by season...
Once the dataset has been created (dataset_init.py)
"""
from utils_tile import Tile
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np

# Progress bar
try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

ROOT = Path(Path(__file__).parents[1], "data")
# Where to save the desired datasets
out_folder = Path(ROOT.parent, "results/tile_extracted_values")

dataset_path = Path(ROOT.parent, "results/dataset.gpkg")
dataset = gpd.read_file(dataset_path)

seasons = {
    "summer": ['06-01', '08-31'],
    "spring": ['03-01', '05-31'],
    "summerlong": ['05-01', '08-31']
}

# Borini Landsat/Sentinel composites images
borini_path = r"H:\Borini\harmoPAF\HarmoPAF_time_series"

# Select only points within image tiles
tile_bboxes_path = Path(ROOT, "tiles_perimeters.gpkg")
tile_bboxes = gpd.read_file(tile_bboxes_path)
tile_bboxes_prj = tile_bboxes.to_crs(dataset.crs)

# Select only the data inside any tile. how=inner remove unmatched points
p_data = dataset.sjoin(tile_bboxes_prj, how="inner", predicate="within")
# Keep all dataset columns plus tile_name data (in order to select the points)
target_cols = dataset.columns.to_list() + ["name"]
# Note: The above operation could yield duplicated values if one point is over
# two different tiles. Remove duplicates by index.
p_data = p_data.loc[~p_data.index.duplicated(keep='first'), target_cols]
p_data.rename(columns={"name": "tile_name"}, inplace=True)

# Retrieve already computed tiles
computed_tiles = [f.stem.split("tile_")[1] for f in out_folder.glob("*.gpkg")]

for tname in tile_bboxes["name"].to_list():

    print(f"Init {tname}")
    # Store the data from each year inside the tile included in the point data
    tdata = []
    if tname in computed_tiles:
        continue

    tile_path = Path(borini_path, tname)
    t = Tile(tile_path)
    
    # Important: must be one tile by name
    pnts = p_data.query(f"tile_name == '{tname}'")
    pnts_years = pd.unique(pnts["YEAR"])
 
    # Filter dates
    t.filter_years(pnts_years, [3, 8])

    if len(t.composite_props.keys()) == 0:
        print("No valid years.")
        continue

    # Gathering the data
    xarr = t.read_xarr()
    
    # Handle nodata value to speed up computations
    nodata_value = xarr.rio.nodata
    # Replace nodata with np.nan
    xarr = xarr.where(xarr != nodata_value, np.nan)

    # Important: point coordinates should be now in the image CRS,
    # in order to extract the correct data from the image
    target_crs = xarr.rio.crs
    pnts = pnts.to_crs(target_crs)
    
    # Deprecated: In previous versions, where a splitted tile geometry was used
    # to filter the points, the following approach filtered the points by
    # sub tile bbox.
    # tile_bbox = tile_bboxes.query(f"name == '{tname}'").to_crs(target_crs)
    # xmin, ymin, xmax, ymax  = tile_bbox.bounds.iloc[0, :].to_list()
    # pnts = p_data.to_crs(target_crs).cx[xmin:xmax, ymin:ymax]

    if pnts.crs != target_crs:
        print("CRS error.")
        continue

    it_message = f"Extracting {tname} data by year"
    iterator = tqdm(pnts_years, desc=it_message) if TQDM else pnts_years
    for year in iterator:
        
        if year not in t.get_years():
            continue

        year_df = pnts.query(f"YEAR == {year}").reset_index(drop=True)

        for sname, sdates in seasons.items():
 
            dates = [np.datetime64(str(year) + '-' +  s) for s in sdates]
            subset = xarr.sel(time=slice(dates[0], dates[1]))

            mean = subset.mean("time", skipna=True) # xarray.DataArray

            # Debugging
            # make sure mean has CRS and transform
            # mean.rio.write_crs(xarr.rio.crs, inplace=True)
            # Set na as nodata value
            # mean.rio.write_nodata(np.nan, inplace=True)
            # export to GeoTIFF
            # tiff_name = Path(out_folder, f"mean_{sname}_{year}.tif")
            # mean.rio.to_raster(tiff_name)

            samples = t.xarr_subtract(year_df, mean)

            season_columns = [b + "_" + sname for b in t.band_names]
            bands_df = pd.DataFrame(samples, columns=season_columns)
            year_df = year_df.join(bands_df)
        
        tdata.append(year_df)

    if len(tdata) == 0:
        continue
    pd.concat(tdata).to_file(Path(out_folder, f"tile_{tname}.gpkg"))
