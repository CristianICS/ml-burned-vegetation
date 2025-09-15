import libpysal
from pathlib import Path
from esda.moran import Moran
from utils_models import Dataset

ROOT = Path(__file__).resolve().parent.parent
# Load your GeoDataFrame
labels_dataset_path = Path(ROOT, "results/dataset_with_values.gpkg")
label_codes_path = Path(ROOT, "data/labels/label_codes.csv")
dataset = Dataset(labels_dataset_path, label_codes_path, 3)

# Choose the numeric variable you want to analyze
variable = dataset.matrix['NDVI_summer']

# Step 1: Create a spatial weights matrix based on distances
# k-nearest neighbors (for example, 5 nearest neighbors)
k = 5
coords = [(point.x, point.y) for point in dataset.matrix.geometry]
w = libpysal.weights.KNN.from_array(coords, k=k)

# Step 2: Row-standardize the weights
w.transform = 'r'

# Step 3: Compute Moran's I
moran = Moran(variable, w)

# Step 4: Print the results
print(f"Moran's I: {moran.I}")
print(f"Expected I: {moran.EI}")
print(f"p-value: {moran.p_sim}")
