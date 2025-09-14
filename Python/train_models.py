from utils_models import Dataset, loop_training
from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle

# Dataset version
version = 3
# Grab folder with stats in non-loop trainings
# When the folder time is not set to None, for loop by predictor ID is skipped
ctime_id = None
use_loop = True

if ctime_id is None:
    # Save time as id for the current training session
    ctime_id = datetime.now().strftime("%Y%m%dT%H%M%S")

ROOT = Path(__file__).resolve().parent.parent

outpath = Path(ROOT, f"results/logs/train_dataset_v{version}_{ctime_id}")
outpath.mkdir(exist_ok=True, parents=True)

# Import labels and apply pretreatments described in inspect_predictors.ipynb
labels_dataset_path = Path(ROOT, "results/dataset_with_values.gpkg")

dataset = Dataset(labels_dataset_path, version)

# print(dataset.predictor_groups())
# Avoid iteration through predictor ids to avoid freezing with dataset 2-3
pred_id = 'LspringPCA'

# Init variables to store the stats and confusion matrices
# Save current predictor set stats, computed with the best gridcv model
stats_path = Path(outpath, f"best_gridcv_stats.csv")
try:
    saved_stats = pd.read_csv(stats_path)
    stats_list = [saved_stats]
    # Check for predictor ID
    if not use_loop and pred_id in pd.unique(saved_stats["pred_id"]):
        raise ValueError(f"Predictor id {pred_id} already exists.")
except:
    stats_list = []

try:
    with open(Path(outpath, f"gridcv_stats.pkl"), "rb") as f:
        grid_list = pickle.load(f)
except:
    grid_list = []

# Save confusion matrices
try:
    with open(Path(outpath, f"confusion_matrices.pkl"), 'rb') as f:
        cm_list = pickle.load(f)
except:
    cm_list = []

if not use_loop:
    print(f"Start model training with {pred_id} dataset")
    cm_dict, stats, grid_dict = loop_training(dataset, pred_id)

    cm_list.append(cm_dict)
    stats_list.append(stats)
    grid_list.append(grid_dict)

else:

    for pred_id in dataset.predictor_sets.keys():

        print(f"Start model training with {pred_id} dataset")
        cm_list, stats_list, grid_list = loop_training(
            dataset, pred_id, cm_list, stats_list, grid_list)

# Save current predictor set stats, computed with the best gridcv model
stats_outpath = Path(outpath, f"best_gridcv_stats.csv")
pd.concat(stats_list).to_csv(stats_outpath, index=False)

# Store gridcv metrics ave dictionary to .pkl file
with open(Path(outpath, f"gridcv_stats.pkl"), 'wb') as fp:
    pickle.dump(grid_list, fp)

# Save confusion matrices
with open(Path(outpath, f"confusion_matrices.pkl"), 'wb') as fp:
    pickle.dump(cm_list, fp)
