# Machine Learning for Post-Fire Vegetation Classification in Aragón

This repository contains the Python code and workflows used in the article:

**Machine Learning Approaches for Temporal Classification of Forest and Shrub Cover in Burned Landscapes: Aragón, Spain**

The framework integrates **Landsat multispectral imagery**, **topographic and edaphic variables**, and **machine-learning classifiers** (Random Forest and Support Vector Machines) to classify forest vegetation and its associated burned-shrub states across Aragón, Spain.

All pipelines are fully reproducible and have been published on [Zenodo](https://zenodo.org/) ![DOI](https://zenodo.org/badge/1056569546.svg).

---

## Features

* Preprocessing of Landsat and ancillary data
* Construction of training datasets from manual, inventory-based, and automatic sources
* Covariate selection using correlation, VIF, and PCA approaches
* Model training with Random Forest and Support Vector Machine
* Resampling strategies for class imbalance (Random Undersampling, Tomek Links, SMOTE, SMOTENC)
* Model evaluation with balanced accuracy, Cohen’s kappa, producer’s and user’s accuracy
* Variable importance analysis with SHAP values
* Assessment of spatial autocorrelation using Moran’s I

---

## Landsat images

Landsat images are stored in a folder for each tile, and the classes inside `utils_tile.py` help manage them. They were produced using the methods described in:

> Alves, D., Fidelis, A., Pérez-Cabello, F., Alvarado, S.T., Estevão Conciani, D., Contursi Cambraia, B., Laffayete Pires da Silveira, A., Freire, T.S., 2022. Impact of image acquisition lag-time on monitoring short-term postfire spectral dynamics in tropical savannas: the Campos Amazônicos Fire Experiment. J. Appl. Remote Sens. 16, 1–51. [https://doi.org/10.1117/1.jrs.16.034507](https://doi.org/10.1117/1.jrs.16.034507)

---

## IFN Data

Due to a confidentiality agreement, the IFN data used to produce dataset versions 2 and 3 is not included.

---

## Workflow

1. `Python/download_dem.py`
   Download the elevation data for each tile and create the related predictor variables.

2. `Python/download_siose.py`
   Save the SIOSE information for each image tile.

3. `Python/extract_soil_siose.py`
   Automatically extract labels from the sparse vegetation class.

4. `Python/create_dataset.py`
   Combine data from manual digitization, sparse vegetation, and IFN into a single file, and add the predictor variables.

5. `Notebooks/inspect_predictors.ipynb`
   Perform data analysis to remove outliers and select the best predictor sets.

6. `Python/train_models.py`
   Train each model pipeline and save the statistics.

7. `Notebooks/inspect_models.ipynb`
   Review the statistics from the model training phase.

8. `Python/moran_i.py`
   Compute Moran's I to assess spatial autocorrelation in the dataset.

---

## Install environments

To run workflow steps 1–7, create the `classification` environment in conda:

```bash
conda env create --file=requirements.yml
```

To run `moran_i.py`, create the `moran` environment:

```bash
conda env create --file=requirements_moran.yml
```
