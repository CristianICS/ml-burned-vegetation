"""
Classify arrays with predictor variables. Two models exist, one with summer and
spring Landsat bands, and another with one summer variables.

Depending on the available data one of the above models will be selected.
"""
from matplotlib import pyplot as plt # type: ignore
from pathlib import Path
import geopandas as gpd # type: ignore
import numpy as np
import rasterio # type: ignore
from utils import Tile, prepare_composites, classify
import traceback
import logging

ROOT = r"D:\iranzo\doctorado_V2\classification"
images_dir = r"H:\Borini\harmoPAF\HarmoPAF_time_series"
output_path = Path(ROOT, "results")

# Load image aois
tiles_path = Path(images_dir, "tiles_perimeters", "tiles_perimeters.gpkg")
tiles_aois = gpd.read_file(tiles_path)

# Load fire perimeters (these are only contained in Aragon)
fires_path = Path(ROOT, "data/fires.gpkg")
fires = gpd.read_file(fires_path, layer="PUFsar").to_crs(tiles_aois.crs)
# Original Year column has str type
fires = fires.astype({'Year': np.int16})
# Filter fires (pre fire more than 1985)
fires = fires.query("Year >= 1986")

# Plot fires plus image aois.
# tile_plot = tiles_aois.plot(figsize=(3,3))
# fires.plot(ax=tile_plot, color="red")
# plt.show()

# Perform a spatial join in order to match fires with tile image dates
fir_tiles = gpd.sjoin(
  fires, tiles_aois, predicate='within', rsuffix="tile").reset_index(drop=True)

# Goal: Select the tile with the maximum data range by fire
# Group fires by tiles_aois `final_year` columns
# Select the IDs where this value is the maximum.
best_idx_tile = fir_tiles.groupby(by=['IDPAF'])['final_year'].idxmax()
fir_best_tiles = fir_tiles.loc[best_idx_tile,]

# Models metadata
models = {
    'summer_only': {
        'predictor_names': ['swir1_summer', 'nir_summer', 'NDVI_summer', 
                            'SHADOW', 'ELEVATION', 'ACIBASI'],
        'src': Path(ROOT,
                    "data/models/predictorid_2_rf_none_adasyn_model_pipe.pkl")
    },
    'summer_spring': {
        'predictor_names': ['swir1_summer', 'swir2_summer', 'NDVI_summer', 
                            'blue_spring', 'nir_spring', 'swir1_spring', 
                            'NDVI_spring', 'SHADOW', 'ELEVATION', 'ACIBASI'],
        'src': Path(ROOT,
            "data/models/predictorid_4_svm_random_soft_none_model_pipe.pkl")
    }
}

# Predictor variable paths
paths_dict = {
    "elev": Path(ROOT, r"data\predictor_variables\dem\dem.vrt"),
    "shadow": Path(ROOT, r"data\predictor_variables\dem\shadow.vrt"),
    "acibasi": Path(ROOT, r"data\predictor_variables\geologia\ACIBASI.tif")
}

# Extract the tile image folders
tile_folders = [t.name for t in Path(images_dir).glob("p*")]

unclassified = {}

# Classify the images, first with summer and spring models
for tile_folder in tile_folders:

    if tile_folder != "p2":
        continue

    print(tile_folder)
    tile = Tile(Path(images_dir, tile_folder))

    # Spring and summer composites
    season_names = ["spring", "summer"]
    season_dates = [['03-01', '05-31'], ['06-01', '08-31']]
    # composites_sprsmm = prepare_composites(tile, season_names, season_dates)
    # Summer composites
    season_names = ["summer"]
    season_dates = [['05-01', '08-31']]
    composites_summer = prepare_composites(tile, season_names, season_dates)

    # Get target fires inside the tile
    tile_fires_i = fir_best_tiles.query(f"name=='{tile_folder}'")

    # Classifying each fire
    for idpaf in tile_fires_i["IDPAF"].to_list():

        print("IDPAF:", idpaf)
        i_fire = fires.query(f"IDPAF == {idpaf}")

        # Classification (without probability, only labels)
        try:
            cls_dict, cls_prob_dict, cls_meta = classify(
                composites_summer, i_fire, paths_dict, models["summer_only"])

            # Create the name for the classified output image
            cls_folder = Path(output_path, f'classified_fires')
            cls_folder.mkdir(exist_ok=True, parents=True)
            cls_file = Path(cls_folder, f'cls_{idpaf}.tif')

            cls_meta.update({
                'compress': 'deflate',
                'predictor': 2,
                'dtype': 'float64',
                'count': len(cls_dict.keys())
            })

            with rasterio.open(cls_file, 'w', **cls_meta) as dst:
                for idx, (name, arr) in enumerate(cls_dict.items()):
                    # Write each band
                    dst.write(arr, idx + 1)
                    # Set band name
                    dst.set_band_description(idx + 1, name)

            # Create the probability of each class by year images
            cls_prob_folder = Path(output_path,
                'classified_prob_fires', f'IDPAF_{idpaf}')
            cls_prob_folder.mkdir(exist_ok=True, parents=True)
            # In each folder by idpaf, store the probability of each year
            # i.e., one image per year.
            for year, prob_dict in cls_prob_dict.items():
                cls_prob_file = Path(cls_prob_folder, f'cls_prob_{year}.tif')

                cls_meta.update({
                    'count': len(prob_dict["band_names"])
                })

                with rasterio.open(cls_prob_file, 'w', **cls_meta) as dst:
                    for i, bandname in enumerate(prob_dict["band_names"]):
                        dst.write(prob_dict['array'][i, :, :], i+1)
                        # Set band name
                        dst.set_band_description(i+1, bandname)


        except Exception as e:
            logging.error(traceback.format_exc())
            
            if tile_folder not in unclassified.keys():
                unclassified[tile_folder] = []
            
            unclassified[tile_folder].append(idpaf)

# TODO: Falta saber por qué el modelo de verano y primavera sale mal.
# Todos los valores en todos los años son el mismo.


print(unclassified)

# Create pre and pos time intervals
# pre_time = [i_fire["Year"].iloc[0] - 3, i_fire["Year"].iloc[0] - 1]
# pos_time = [i_fire["Year"].iloc[0] + 9, i_fire["Year"].iloc[0] + 10]