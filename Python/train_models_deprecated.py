from utils_models import Dataset, Model, Pipeline
from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle

# Dataset version
version = 2

ROOT = Path(__file__).resolve().parent.parent

# Save time as id for the current training session
ctime_id = datetime.now().strftime("%Y%m%dT%H%M%S")

outpath = Path(ROOT, f"results/logs/train_dataset_v{version}_{ctime_id}")
outpath.mkdir(exist_ok=True, parents=True)

# Import labels and apply pretreatments described in inspect_predictors.ipynb
labels_dataset_path = Path(ROOT, "results/dataset_with_values.gpkg")

dataset = Dataset(labels_dataset_path, version)

# Init variables to store the stats and confusion matrices
cm_list = []
grid_list = []
stats_list = [] # Store pandas dataframes

for pred_id, pred_vars in dataset.predictor_sets.items():

    print(f"Start model training with {pred_id} dataset")
    X_train, X_test, y_train, y_test = dataset.split(pred_id, "ESPE_rc")

    # Data augmentation techniques
    for da in ["smote", "none"]:
        # Data undersampling techniques
        for du in ["tomeklinks", "random", "none"]:

            pipe_name = f"{da}_{du}"
            print(f"  {pipe_name}")

            # Construct pipeline
            if ("C" in pred_vars) and (da == "smote"):
                pipe = Pipeline(
                    y_train,
                    du,
                    da,
                    X_train,
                    categorical_predictors=pred_vars["C"])
            else:
                pipe = Pipeline(y_train, du, da)

            for model_key in ["rf", "svm"]:
                print(f"    {model_key.upper()}")
                # Apply the grid search cv
                model = Model(model_key)
                pipe.add_model(model.get_clf())
                gridcv, cv = pipe.grid_search(model.get_grid())
                gridcv.fit(X_train, y_train)

                cm_dict, stats = model.compute_metrics(gridcv, X_test, y_test, pipe_name, pred_id)

                # Clean the best params dictionary, remove "clf__" prefix
                grid_best = gridcv.best_params_
                grid_best = {k.removeprefix("clf__"): v for k, v in grid_best.items()}
                # Save the gridcv stats for the current model
                grid_dict = {
                    'pred_id': pred_id,
                    'model': model_key,
                    'pipe': pipe_name,
                    'best_params': grid_best,
                    'best_score': gridcv.best_score_
                }
                cm_list.append(cm_dict)
                stats_list.append(stats)
                grid_list.append(grid_dict)

# Save current predictor set stats, computed with the best gridcv model
stats_outpath = Path(outpath, f"best_gridcv_stats.csv")
pd.concat(stats_list).to_csv(stats_outpath, index=False)

# Store gridcv metrics ave dictionary to .pkl file
with open(Path(outpath, f"gridcv_stats.pkl"), 'wb') as fp:
    pickle.dump(grid_list, fp)

# Save confusion matrices
with open(Path(outpath, f"confusion_matrices.pkl"), 'wb') as fp:
    pickle.dump(cm_list, fp)
