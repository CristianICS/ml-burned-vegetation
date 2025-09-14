import geopandas as gpd
import libpysal
from esda.moran import Moran

# Load your GeoDataFrame
gdf = gpd.read_file("your_points.shp")

# Choose the numeric variable you want to analyze
variable = gdf['your_numeric_column']

# Step 1: Create a spatial weights matrix based on distances
# k-nearest neighbors (for example, 5 nearest neighbors)
k = 5
coords = [(point.x, point.y) for point in gdf.geometry]
w = libpysal.weights.KNN.from_array(coords, k=k)

# Step 2: Row-standardize the weights
w.transform = 'r'

# Step 3: Compute Moran's I
moran = Moran(variable, w)

# Step 4: Print the results
print(f"Moran's I: {moran.I}")
print(f"Expected I: {moran.EI}")
print(f"p-value: {moran.p_sim}")
