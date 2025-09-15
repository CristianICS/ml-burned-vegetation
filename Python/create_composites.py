"""Stack Landsat image together depend on the time interval by fire."""
from utils_tile import Tile, Rasters
from pathlib import Path
import geopandas as gpd
import rasterio
import numpy as np

ROOT = r"D:\iranzo\doctorado_v2\classification"
# Borini time series Landsat/Sentinel images location
borini_path = r"H:\Borini\harmoPAF\HarmoPAF_time_series"
# Folder to save the results
out_path = Path(ROOT, r"data\predictor_variables\landsat")
# Bounding Box of the Borini images
tiles = gpd.read_file(Path(borini_path, "tiles_perimeters/tiles_perimeters.gpkg"))

# Include the fires inside PaF database
fires = gpd.read_file(Path(ROOT, "data/fires.gpkg")).to_crs(tiles.crs)
fires["Year"] = fires["Year"].astype("int16")

# Plot fires plus image aois.
# tile_plot = tiles_aois.plot(figsize=(3,3))
# fires.plot(ax=tile_plot, color="red")
# plt.show()

# Get only fires inside tiles bounding boxes
fires_tiles = gpd.sjoin(
  fires, tiles, predicate='within', rsuffix="tile").reset_index(drop=True)

# Goal: Select the tile with the maximum data range by fire
# Group fires by tile's "final_year" column
# Select the IDs where this value is the maximum.
best_idx_tile = fires_tiles.groupby(by=['IDPAF'])['final_year'].idxmax()
fires_best_tiles = fires_tiles.loc[best_idx_tile,]

# Remove the fires without pre fire image (the ones in 1984, the first year
# with Landsat images in the series)
fires_best_tiles = fires_best_tiles.query("Year > 1985")

seasons = {
    "spring": ['03-01', '05-31'],
    "summer": ['06-01', '08-31'],
    "summerlong": ['05-01', '08-31']
}

def reduce_by_season(tile: Tile, year, dates, season_name, bbox, bbox_crs):
    """
    Execute a composite for the year and dates included in the function.

    :year: Year to perform the composite.
    :dates: List with init and final composite month and day (mm-dd)
    """
    # Filter the images inside the tile
    filter_dates = [f'{int(year)}-{s}' for s in dates]
    tile_meta = tile.filter_date(filter_dates[0], filter_dates[1])
    
    # Interrupt the process due to there are no images
    if len(tile_meta.keys()) == 0:
        return (False, False, False)

    # Open the images from the season
    try:
        images = Rasters(tile_meta, tile.imgs_dir, bbox, bbox_crs)
    except ValueError as e:
        raise ValueError(f"{e}: {season_name} in {year}")

    # Perform the mean
    images.compute_mean()

    # Add a prefix with the season to all img band names to distinguish them
    # new_bnames = [f'{b}_{season_name}' for b in images.composite_band_names]
    return (images.composite, images.band_names, images.composite_meta)

# Composite by fire, year and season
for i, f in fires_best_tiles.iterrows():

    out_dir = Path(out_path, f"IDPAF_{f["IDPAF"]}")
    out_dir.mkdir(exist_ok=True)

    # Construct the path of the tile with its name
    tile = Tile(Path(borini_path, f["name"]))
    # The original perimeters have a high spatial resolution,
    # generalize them in order to speed up the process.
    aoi_buffer = fires.query(f"IDPAF=={f["IDPAF"]}").simplify(
        100, preserve_topology=False)
    # Construct a buffer to include control zone (unburned)
    # Its bounding box will be the area to extract the information.
    aoi = aoi_buffer.geometry.buffer(300).total_bounds

    # Compute composites by year
    for y in tile.get_years():

        for season, dates in seasons.items():

            arr, bnames, meta = reduce_by_season(
                tile, y, dates, season, aoi, fires.crs)

            # When no spring or summer images exists, compute summer long
            if type(arr) == type(False):
                print("No images:", season)
                continue

            # Convert array to integer dtype
            arr = arr.astype(np.int32)

            meta.update(
                # The smallest possible value for 32-bit signed integers
                nodata = np.iinfo(np.int32).min
            )

            # Save composite
            out_name = f"Y{y}_{season}.tif"
            with rasterio.open(Path(out_dir, out_name), "w", **meta) as dst:
                for i, bandname in enumerate(bnames):
                    dst.write(arr[i, :, :], i+1)
                    # Set band name
                    dst.set_band_description(i+1, bandname)
