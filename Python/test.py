from pathlib import Path
from utils_models import Dataset

ROOT = Path(__file__).absolute().parent.parent

# Import labels and apply pretreatments described in inspect_predictors.ipynb
labels_dataset_path = Path(ROOT, "results/dataset_with_values.gpkg")

dataset = Dataset(labels_dataset_path, 1)

pred_id = "LseasonPCA_gHSA"

X_train, X_test, y_train, y_test = dataset.split(pred_id, "ESPE_rc")
print(X_train.columns)