"""
Join labels inside a unique dataset

1. Andrea Acosta / Fernando Pérez labels
2. IFN
3. LUCAS
4. Siose
"""
from pathlib import Path
from matplotlib import pyplot as plt
import geopandas as gpd
import pandas as pd
import urllib

ROOT = Path(Path(__file__).parents[1], "data")
# Where to save the desired dataset
dataset_output = Path(ROOT.parent, "results/dataset.gpkg")

# Five target columns to select
target_cols = ["ESPE", "ESPE_rc", "YEAR", "source", "FCC", "Ocu1", "geometry"]
# Save extracted labels for all the sources
labels = []

# Manually digitized labels (correct format)
manual = gpd.read_file(Path(ROOT, "labels/acosta_labels.gpkg"))
manual["source"] = "Digitized"
labels.append(manual)

# Handle NFI labels
# Include the reclassified ones (ESPE_rc)
codes = pd.read_csv(Path(ROOT, "labels/label_codes.csv"))
def extract_rc(df):
    return df.join(codes.set_index("code_v1"), on="ESPE")["code_v1_reclass"]

for i in [2, 3, 4]:
    ifn = gpd.read_file(Path(ROOT, f"labels/ifn{i}_labels.gpkg"))
    ifn["ESPE_rc"] = extract_rc(ifn)
    if "source" not in ifn.columns:
        ifn["source"] = "NFI" + str(i)

    labels.append(ifn[target_cols])

# Handle soil labels
bare_soil = gpd.read_file(Path(ROOT, "labels/ground_points_db.gpkg"))
bare_soil[["ESPE", "ESPE_rc"]] = 20
bare_soil["source"] = "SIOSE"
bare_soil[["Ocu1", "FCC"]] = 0
labels.append(bare_soil[target_cols].to_crs(manual.crs))

# Add soil data from LUCAS (*Land Use-Land Cover Area Frame Survey*)
# [@rLUCASCover200620182022].
lucas = pd.read_csv(Path(ROOT, r"lucas\LUCAS_2018_Copernicus_attributes.csv"),
                    low_memory=False)
# Open lucas database with geopandas
x_col = lucas['GPS_LONG']
y_col = lucas['GPS_LAT']
g = gpd.points_from_xy(x_col, y_col, crs="EPSG:4326")
lucas = gpd.GeoDataFrame(lucas, geometry=g)
# Get points over Aragon
lucas_aragon = lucas[lucas['NUTS2'] == 'ES24']
# print("LUCAS labels over the region")
# print(lucas_aragon[['CPRN_LC_LABEL','CPRN_LC']].value_counts())

# Select only "Other bare soil" class
lucas_aragon = lucas_aragon.query('CPRN_LC == \'F4\'')
lucas_aragon[['ESPE', 'ESPE_rc']] = 20
lucas_aragon["source"] = "LUCAS"
lucas_aragon[["Ocu1", "FCC"]] = 0
labels.append(lucas_aragon[target_cols])

# Show the selected LUCAS labels in Aragon
# imgs = lucas_aragon['FILE_PATH_GISCO_NORTH'].to_list()

# fig, axs = plt.subplots(1, 3, figsize=(10,5))

# for i in range(3):
#     # Create a file-like object from the url
#     f = urllib.request.urlopen(imgs[i])
#     # read the image file in a numpy array
#     a = plt.imread(f, format='jpg')
#     axs[i].imshow(a)
#     axs[i].set_axis_off()


# Once the dataset is create, add predictor variables information
dataset = pd.concat(labels).reset_index(drop=True)
if not dataset_output.exists():
    dataset.to_file(dataset_output, index=False)
