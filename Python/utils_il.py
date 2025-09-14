"""Create incident local angle (IL) image with GEE scene image parameters."""
from rasterio.warp import transform_bounds # type: ignore
from rasterio.crs import CRS # type: ignore
from statistics import mean
import numpy as np # type: ignore
import rasterio # type: ignore
import math
import ee # type: ignore

ee.Initialize(project="s-correction")

class IL:

    def __init__(self, tile_meta, bounds, crs):
        self.gee_collections = {
            'L05': "LANDSAT/LT05/C02/T1_L2",
            'L07': "LANDSAT/LE07/C02/T1_L2",
            'L08': "LANDSAT/LC08/C02/T1_L2"
        }
        self.props = ['SUN_AZIMUTH', 'SUN_ELEVATION']

        self.prop_vals = self._extract_scene_metadata(tile_meta, bounds, crs)

    def _extract_scene_metadata(self, tile_meta, bounds, crs):
        """
        Extract scene data to compute IL

        The final extracted properties are an average from all the properties
        in the image collection. This approach reduce the border effects
        between tiles due to the different image acquisition parameters.
        
        :tile_meta: utils.Tile.meta
        """
        # Extract bounding box and transform it to global CRS
        # in order to filter collection with it in GEE.
        xmin, ymin, xmax, ymax = bounds
        bounds_4326 = transform_bounds(crs, CRS.from_epsg(4326),
            left=xmin,
            bottom=ymin,
            right=xmax,
            top=ymax)
        xmin, ymin, xmax, ymax = bounds_4326

        # Extract key image properties with GEE
        region = ee.Geometry.BBox(xmin, ymin, xmax, ymax)

        # For each sensor, extract all the image props from the dates
        prop_dict = {p: [] for p in self.props}

        for s in self.gee_collections.keys():
            dates = [v["date"] for v in tile_meta.values() if v["sensor"] == s]
            # Transform dates to valid gee type
            img_dates = [d.strftime("%Y-%m-%d") for d in sorted(dates)]
            
            # When there is only one image for current sensor, generate
            # a final date by adding another date one month ahead
            if len(dates) == 1:
                final_date = ee.Date(img_dates[0]).advance(1, 'month')
            elif len(dates) > 1:
                final_date = img_dates[len(img_dates) - 1]
            elif len(dates) == 0:
                continue
            # Images have been sorted to extract the earliest and latest ones
            # # Init the image collection to obtain the above images
            gee_col = (ee.ImageCollection(self.gee_collections[s])
                .filterDate(img_dates[0], final_date)
                .filterBounds(region)
            )

            for prop in self.props:
                prop_dict[prop] += gee_col.aggregate_array(prop).getInfo()

        # Perform the average of the desired props
        return {k: mean(v) for k, v in prop_dict.items()}

    def open_img(self, img_path):
        """Create a temporal array from image path."""
        with rasterio.open(img_path) as src:
            return src.read(1)
    
    def compute(self, aspect_path, slope_path):
        """
        Compute IL image: Incident local angle

        :aspect: Rasterio numpy.array with aspect values (orientations)
        :slope: Rasterio numpy.array with slope values (degrees)
        """
        sun_az = self.prop_vals['SUN_AZIMUTH']
        sun_el = self.prop_vals['SUN_ELEVATION']
        # Open raster and save it as numpy array
        aspect = self.open_img(aspect_path)
        slope = self.open_img(slope_path)
        # Transformaar los angulos en radianes
        aspect_r = aspect * 0.01745
        slope_r = slope * 0.01745
        # Sun zenithal angle
        ze_sol_r = (90 - sun_el) * 0.01745
        az_sol_r = sun_az * 0.01745
        # Generar funciones para calcular el coseno y el seno
        cos = lambda x: abs(math.cos(x))
        sin = lambda x: abs(math.sin(x))
        # Calcular el angulo de incidencia local
        il = cos(ze_sol_r) * np.cos(slope_r) + sin(ze_sol_r) * np.sin(slope_r) * np.cos(az_sol_r - aspect_r)
        # La imagen de iluminacion es en realidad el coseno del angulo de
        # incidencia local. No es necesario realizar np.cos(il).
        return il
