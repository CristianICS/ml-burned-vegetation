# Clasificación Machine Learning

Objetivo: clasificar vegetación forestal y su subtipo de matorral alrededor de incendios forestales.

Los píxeles que hayan sido quemados tras un incendio pasarán a considerarse matorral de la vegetación preexistente al elaborar las etiquetas.

Estudiar cómo varía el acierto global y por categorías según:

- Tipo de clasificador (aplicar gridsearchcv)
- Tipo de preprocesamiento (secuencias de aumento y disminución)
- Número de etiquetas incluidas
- Número de variables predictoras

## Variables predictoras

Se componen de las bandas de imágenes Landsat harmonizadas (con las correcciones necesarias para reducir las diferencias espaciales entre los píxeles de toda la serie). Existen imágenes desde 1984 hasta 2023.

El procedimiento de cálculo es costoso, por lo que solo se calcularon datos para las zonas quemadas de Aragón (`data/divided_tiles_by_area.gpkg`).

La clasificación regional de la vegetación a una escala espacial de 30 metros (píxeles Landsat) presenta problemas complejos. Las firmas de la vegetación se mezclan en píxeles tan grandes, por lo que es más difícil diferenciar entre tipos similares de vegetación. Es por ello que se han utilizado variables con información adicional con el objetivo de complementar la información espectral.

Se dividen en dos tipos. En primer lugar, las derivadas del Modelo Digital de Elevaciones obtenido de los servicios WCS del PNOA. Se selecciona el ráster de 25 metros de resolución espacial en EPSG:25830. La extracción se realiza utilizando las zonas que cuentan con imágenes Landsat disponibles.

De las alturas se derivan:

- [Pendientes](https://gdal.org/en/stable/programs/gdal_raster_slope.html), en grados.
- Orientaciones [https://gdal.org/en/stable/programs/gdal_raster_aspect.html], entre 0 y 360 grados. Representan el el acimut (azimuth) hacia el cual se orientan las pendientes. El valor será `nodata` si el valor de la pendiente es 0.
    - 0: Norte
    - 90: Este
    - 180: Sur
    - 270: Oeste
- Sombras ([Hillshade](https://gdal.org/en/stable/programs/gdal_raster_hillshade.html)), utilizando un acimut de 180 grados (sur) y una elevación de 45 grados (sol de mediodía).

Además, se ha obtenido la variable ácido/básico, capa de información binaria derivada del mapa geológico de Aragón (1993). La división entre rocas básicas y ácidas se realiza con el objetivo de mejorar la clasificación de la especie *Pinus pinaster*, con preferencia por sustratos básicos. A los códigos seleccionados como rocas básicas se les asigna un 1, y a los clasificados como ácidos un 2.

## Sets de etiquetas

El conjunto de etiquetas se ha obtenido en distintos años de la serie temporal de imágenes Landsat. Por este motivo, algunas etiquetas mantienen su posición a lo largo de varios años. Sin embargo, si han sido observadas en varios años tendrán dos valores, uno por cada año, y los valores de las variables predictoras tendrán en cuenta dicha fecha para obtener las imágenes Landsat.

### Vegetación rala

Todos los conjuntos de etiquetas utilizados contienen datos de zonas con un porcentaje elevado de suelo desnudo mezclado con vegetación rala o dispersa. Se utilizan los códigos de python `extract_ground_points.py` y las funciones en `utils_il.py`. Además, el código `create_dem.py` debe de haber sido utilizado para crear los MDEs con los que se obtiene la información de pendiente y orientaciones necesarias para la creación de la imagen de Iluminación. También deben descargarse las escenas del SIOSE con el código "siose.py".

> Nota: Se han puesto fuera del archivo `utils.py` debido a que no se quiere abrir el módulo `gee` cada vez que se utilicen estas funciones.

Procedimiento:

1. Seleccionar los años del SIOSE (2005, 2009, 2011 y 2014)
2. Crear un compuesto con imágenes Landsat entre el 1 de junio y el 31 de julio en cada año con información del SIOSE disponible.
3. Seleccionar los píxeles con un valor de NDVI comprendido entre 0.08 y 0.15. Estos serán los píxeles candidatos que podrían ser vegetación rala.
4. Filtrar los puntos por distancia, eliminando aquellos con puntos a menos de 200 metros.
5. Obtener una imagen con el coseno del ángulo de incidencia local (IL) utilizando los datos de Azimuth y Elevación solar promedio de las escenas utilizadas en el compuesto del NDVI.
6. Seleccionar de entre los píxeles candidatos aquellos con un valor IL superior a 0.7.
7. Abrir la escena del SIOSE correspondiente, descargada del servicio WMS de lDEE. Se seleccionan aquellos píxeles candidatos sobre categorías de ocupación del suelo "roquedo" y "suelo desnudo".

### Digitalización manual

El primer set de etiquetas fue definido por Andrea Acosta y Fernando Pérez Cabello. Cuenta con 3132 etiquetas definidas. Se le agregaron puntos del IFN para poder aumentar las etiquetas.


1. Mejor método de pretratamiento

  - `RandomUnderSampler` vs `TomeKLinks` vs `None`
  - `SMOTE` vs `ADASYN` vs `None`


## Creación de los conjuntos de datos



## Modelos

Nota: [La librería Scikit-Learn no puede ejecutarse en una GPU](https://stackoverflow.com/a/41568439). Es posible si se utiliza [cuml](https://docs.rapids.ai/api/cuml/stable/cuml_intro/), pero no lo he testado todavía.

## Aumentar la eficiencia del código

Gracias por describir tan bien tu flujo 👌. Entiendo por qué te está tardando tanto: estás haciendo muchas operaciones costosas (lectura de múltiples imágenes muy grandes, reproyecciones y accesos píxel a píxel). Te paso varias sugerencias concretas para **mejorar la eficiencia** de tu proceso en Python con Rasterio y datos satelitales:

1. **Reducir accesos píxel a píxel (lo más costoso)**

   * Evita recorrer los puntos uno a uno con `.index(row, col)` y `.read()` en cada iteración.
   * Usa `rasterio.sample()` o mejor aún **`rasterio.mask`** o **`rasterstats`** (o `rioxarray`) que permiten extraer todos los valores de un conjunto de puntos en bloque.
   * Otra opción: transformar las coordenadas de todos los puntos a filas/columnas de una vez (vectorizado con NumPy), y luego indexar directamente los arrays cargados.

---

2. **Lectura más eficiente de los rásteres**

   * Si trabajas con miles de archivos (1985–2023), evita abrir/cerrar con `rasterio.open()` en cada bucle.
     👉 Usa `rioxarray.open_rasterio()` (que da un objeto `xarray.DataArray`) y permite trabajar con **chunks de dask** → esto paraleliza y carga bajo demanda.
   * Considera **apilar las imágenes en un solo stack** (NetCDF, Zarr o Cloud Optimized GeoTIFF). Así no abres 1000 archivos, sino que consultas una sola estructura optimizada.

---

3. **Optimizar las subáreas**

   * Ahora divides en "subáreas de 20 mil hectáreas".
     En lugar de eso, puedes **hacer un índice espacial** (por ejemplo con `geopandas.sjoin`) para relacionar cada punto con su tile/subárea de forma previa. Así reduces cálculos repetidos de ventanas.
   * Si sigues usando ventanas: asegúrate de aprovechar `rasterio.windows.from_bounds` con NumPy slicing, que es bastante rápido.

---

4. **Promedios por estación/año**

   * No cargues imágenes una por una para calcular medias.
     Usa `xarray` + `groupby("time.season")` + `.mean("time")` → esto calcula todo de golpe, con paralelización.
   * Incluso si sigues con rasterio puro: mejor leer todos los arrays de esa estación en un stack `np.array()` y aplicar `.mean(axis=0)` en memoria (si cabe).

---

5. **Paralelización**

   * Tu flujo es muy paralelizable: por **tile**, **subárea** o incluso **año/estación**.
     Usa `concurrent.futures.ProcessPoolExecutor` o `dask.distributed` para explotar varios cores.
   * Si cada tile tarda 2–3 h, correr 10 tiles en paralelo en un servidor de 16–32 cores puede reducir drásticamente el tiempo.

---

6. **Formatos de archivo**

   * Si las imágenes son GeoTIFF clásicos, conviértelas a **Cloud Optimized GeoTIFF (COG)** → mucho más rápido para lecturas por ventanas (`window`).
   * Considera precalcular un **data cube** (ej. en formato Zarr). Zarr + Dask es prácticamente estándar para series largas de imágenes satelitales.

---

7. **Bibliotecas especializadas**

   * \[`xarray` + `rioxarray` + `dask`] → mejor combinación para time series + raster.
   * \[`geopandas` + `rasterstats`/`xarray-spatial`] → extracción en puntos más eficiente.
   * \[`pyogrio`] (más rápido que fiona para leer vectores).

---

## ⚡ Ejemplo de enfoque más eficiente (esquema con xarray)

```python
import rioxarray
import geopandas as gpd

# 1. Cargar puntos
points = gpd.read_file("puntos.gpkg")

# 2. Abrir stack de imágenes con dask
stack = rioxarray.open_rasterio(
    "imagenes.zarr",  # o varios COGs con `open_mfdataset`
    chunks={"x": 1024, "y": 1024, "time": 1}
)

# 3. Agrupar por estación y año
seasonal_means = stack.groupby("time.season").mean("time")

# 4. Extraer valores en puntos
# rioxarray tiene un método .sel_points / o usar rasterio.sample en bloque
vals = [stack.sel(x=p.x, y=p.y, method="nearest").values for p in points.geometry]
```

Con `dask`, los cálculos se hacen en paralelo y sólo se leen los fragmentos de disco necesarios.

---

👉 Resumen de lo más efectivo para ti:

1. **No recorrer puntos uno a uno** → usa `sample` o indexación vectorizada.
2. **Agrupa y calcula medias en bloque** (no en bucles).
3. **Convierte a COG o Zarr y usa xarray+dask** para lectura eficiente y paralela.
4. **Paraleliza por tile/año** en varios procesos.

---

```python
# Filtrar entre 2000-01-01 y 2010-12-31
subset = stack.sel(time=slice("2000-01-01", "2010-12-31"))

# Media en ese rango
mean_interval = subset.mean("time")
```

En lugar de guardar los archivos con zarr, que duplicaría la información, existen dos opciones más interesantes. Abrir los datos con xopenarray por season o crear Cloud Optimized Geotiffs.

💯 Tu preocupación es muy válida:
cuando pasas de miles de **GeoTIFFs** a un **Zarr “stackeado”**, efectivamente estás **duplicando todo** el volumen de datos en disco, y en satélite eso puede ser teras.

La clave está en **no duplicar**, sino construir un **catálogo / índice** que apunte a los archivos existentes y permita a `xarray+dask` tratarlos como si fueran un solo “data cube”.

---

## 🔑 Alternativas a escribir un Zarr duplicado

### 1. **Abrir directamente los TIFFs con `open_mfdataset`**

`xarray` puede abrir múltiples TIFFs (usando `rioxarray.open_rasterio`) y tratarlos como un dataset unificado.

```python
import xarray as xr
import rioxarray
import glob
import re
from datetime import datetime

files = sorted(glob.glob("/ruta/tile_001/*.tif"))

# Extraer fechas desde nombres
def get_time(fname):
    match = re.search(r"(\d{8})", fname)
    return datetime.strptime(match.group(1), "%Y%m%d")

times = [get_time(f) for f in files]

# Abrir como un solo dataset sin escribir nada nuevo
rasters = [rioxarray.open_rasterio(f, chunks={"x": 1024, "y": 1024}).expand_dims(time=[t])
           for f, t in zip(files, times)]

stack = xr.concat(rasters, dim="time")
```

👉 Aquí **no duplicas los datos**: cada TIFF se sigue leyendo desde su archivo original, pero `xarray+dask` los maneja como un stack.

---

### 2. **Usar Cloud Optimized GeoTIFF (COG)**

Si conviertes tus TIFF a **COG** (sólo una vez, reemplazando los originales), no necesitas Zarr:

* COG ya permite acceso por bloques (ventanas).
* Dask los gestiona muy bien.

Herramientas: `gdal_translate -of COG input.tif output.tif`

---

### 3. **Crear un STAC catalog / Intake catalog**

En vez de un “stack físico”, puedes crear un **catálogo de metadatos** (STAC, Intake, even JSON+CSV) que describe:

* ruta del archivo
* fecha
* tile
* bandas disponibles

Después, `intake-stac` o `xarray` pueden leer directamente esos TIFFs “on demand”.
Esto es lo que usan plataformas como Pangeo o Earth Engine: no duplican, sólo indexan.

---

### 4. **Virtual stacking con `Rasterio VRT`**

Puedes generar un **VRT (Virtual Raster Tile)**, que es básicamente un XML con la lista de tus TIFFs apilados.
Se crea con:

```bash
gdalbuildvrt -separate stack.vrt *.tif
```

Ese `.vrt` ocupa unos KB y actúa como un mosaico/stack virtual.
Luego puedes abrir `stack.vrt` con `rioxarray` y tratarlo como un multibanda.

👉 Esto es súper útil si lo que quieres es **unificar imágenes** sin copiarlas.

---

## 📌 Comparación rápida

| Método                     | Duplica datos        | Rendimiento                    | Portabilidad                    |
| -------------------------- | -------------------- | ------------------------------ | ------------------------------- |
| **Zarr**                   | ✅ Sí                 | 🚀 Muy alto                    | Python/Dask                     |
| **open\_mfdataset (lazy)** | ❌ No                 | ⚡ Alto (depende de TIFF)       | Python                          |
| **COG**                    | ❌ No (si reemplazas) | 🚀 Muy alto                    | Estándar en ecosistema satélite |
| **STAC/Intake**            | ❌ No                 | ⚡ Alto (on demand)             | Multiplataforma                 |
| **VRT**                    | ❌ No                 | ⚡ Alto (GDAL depende de TIFFs) | GDAL, Rasterio                  |

---

## ✅ Recomendación práctica

* Si **no tienes espacio** para un segundo dataset → usa **VRT** o **open\_mfdataset** (no duplican, solo indexan).
* Si después quieres mover tu flujo a la nube o clusters → considera migrar a **COG + STAC** (estándar abierto).

---

👉 Pregunta: ¿quieres que te prepare un ejemplo concreto con **GDAL VRT** (que unifica automáticamente todos los TIFF de cada carpeta en un stack virtual que luego lees con `xarray`), o prefieres quedarte en el flujo con `xarray.open_mfdataset` sin generar archivos adicionales?

