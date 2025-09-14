from pathlib import Path
from collections import Counter
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.preprocessing import FunctionTransformer # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
from sklearn.model_selection import StratifiedKFold # type: ignore
from imblearn.under_sampling import RandomUnderSampler, TomekLinks # type: ignore
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC  # type: ignore
from imblearn.pipeline import Pipeline as imbPipe# type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.metrics import balanced_accuracy_score # type: ignore
from sklearn.metrics import cohen_kappa_score # type: ignore
from sklearn.metrics import confusion_matrix # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from matplotlib.collections import QuadMesh # type: ignore
import matplotlib.font_manager as fm # type: ignore
import geopandas as gpd
import seaborn as sn # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import warnings


pd.options.mode.copy_on_write = True

def bg_undersampling(df):
    """Perform a reduction in the very high bare ground number of labels"""
    X = df.drop(columns=['code_v1_reclass'])
    y = df['code_v1_reclass']

    # Find the sparse vegetation class
    max_clss = y.value_counts().idxmax()
    # Define the undersampler
    rus = RandomUnderSampler(
        sampling_strategy={max_clss: 800},
        random_state=42
    )

    # Apply undersampling
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # Create new dataframe. This is mandatory to remove NA values in each
    # predictor set iteration and remove the ESPE codes related to them
    dataset = pd.concat([
        X_resampled.reset_index(drop=True),
        y_resampled.reset_index(drop=True)],
        axis=1
    )

    return dataset

def drop_outliers(df):
    target = df.copy()
    # Likely water, snow, clouds, or some artificial surfaces.
    target.query("(NDVI_summer > 0) and (NDVI_spring > 0)", inplace=True)
    # Likely clouds or Landsat 7 errors
    target.query("swir1_spring < 9000", inplace=True)
    return target

class Dataset:

    def __init__(self, labels_dataset_path, version):
        """Preprocess the label data to conduct the training phase."""
        # Define dataset versions
        dat_queries = {1: "(source == 'Digitized') | (source == 'SIOSE')"}
        dat_queries[2] = f"(FCC > 20) & (Ocu1 > 4) | ({dat_queries[1]})"
        dat_queries[3] = f"(FCC > 20) & (Ocu1 > 6) | ({dat_queries[1]})"
        
        # Import pretreatments described in inspect_predictors.ipynb
        dataset = gpd.read_file(labels_dataset_path)
        # Drop Pinus uncinata
        dataset.query(
            "(code_v1_reclass != 5) & (code_v1_reclass != 15)", inplace=True)
        # Add the target data by version
        dataset.query(dat_queries[version], inplace=True)

        # Reduce the abnormal amount of bare ground labels
        dataset = bg_undersampling(dataset)
        
        # Store columns containing predictor variables
        global_pred = ["dem", "shadow", "slope", "acibasi"]
        pred_vars = (
            dataset.columns.str.endswith("summer")
            | dataset.columns.str.endswith("spring")
            | dataset.columns.isin(global_pred)
        )
        # Handle NA values (0) in the above columns
        dataset.loc[:, pred_vars] = (
            dataset.loc[:, pred_vars]
            .where(dataset.loc[:, pred_vars] >= 0, 0)
        )
        # Remove columns with nodata values
        dat_i = dataset[(dataset.loc[:, pred_vars] != 0).all(axis=1)]

        # Create NDVI variables
        dat_i.loc[:, "NDVI_summer"] = self.ndvi(dat_i, "summer")
        dat_i.loc[:, "NDVI_spring"] = self.ndvi(dat_i, "spring")

        # Remove outlayers
        self.matrix = drop_outliers(dat_i)

        self.define_predictor_groups(dat_i.columns)

    def ndvi(self, df, suffix):
        numerator = df[f"nir_{suffix}"] - df[f"red_{suffix}"]
        denominator = df[f"nir_{suffix}"] + df[f"red_{suffix}"]
        return numerator / denominator

    def define_predictor_groups(self, cols):
        """
        Predictor variables dictionary

        PCA = Variables to reduce, N = Numerical, C = Categorical

        In addition, the parameter "remove_pca_cols" exclude tha reduced columns
        from the output.
        """
        pred_dict = {}
        
        # Dataset 1
        pred_dict[1] = {
            ["nir_summer", "swir1_spring", "NDVI_summer"]
        }

        pred_dict['LsummerPCA']= {
            "PCA": {
                "summer": list(cols[cols.str.endswith("summer")])
            },
            "remove_pca_cols": True
        }

        pred_dict['LsummerPCA_gHSA'] = {
            "PCA": pred_dict["LsummerPCA"]["PCA"],
            "N": ["dem", "shadow"],
            "C": ["acibasi"],
            "remove_pca_cols": True
        }

        pred_dict['LspringPCA']= {
            "PCA": {
                "spring": list(cols[cols.str.endswith("spring")])
            },
            "remove_pca_cols": True
        }

        pred_dict['LspringPCA_gHSA'] = {
            "PCA": pred_dict["LspringPCA"]["PCA"],
            "N": ["dem", "shadow"],
            "C": ["acibasi"],
            "remove_pca_cols": True
        }

        pred_dict["LseasonPCA_gHSA"] = {
            "PCA": {
                "spring": list(cols[cols.str.endswith("spring")]),
                "summer": list(cols[cols.str.endswith("summer")])
            },
            "N": pred_dict['LspringPCA_gHSA']["N"],
            "C": pred_dict['LspringPCA_gHSA']["C"],
            "remove_pca_cols": True
        }

        pred_dict["LPCA_gHSA"] = {
            "PCA": {
                "all": (
                    list(cols[cols.str.endswith("spring")]) +
                    list(cols[cols.str.endswith("summer")])
                )
            },
            "N": pred_dict['LspringPCA_gHSA']["N"],
            "C": pred_dict['LspringPCA_gHSA']["C"],
            "remove_pca_cols": True
        }

        pred_dict["PCA"] = {
            "PCA": {
                "all": (
                    pred_dict["LPCA_gHSA"]["PCA"]["all"] +
                    pred_dict['LspringPCA_gHSA']["N"] +
                    pred_dict['LspringPCA_gHSA']["C"]
                )
            },
            "remove_pca_cols": True
        }

        # Setting predictors by spectral signatures plot by species
        pred_dict["Ls_v1"] = {
            "N": [
                    "nir_spring", "nir_summer", "swir1_spring", "swir1_summer",
                    "swir2_summer", "swir2_spring", "red_summer", "red_spring"
                ],
            "remove_pca_cols": True
        }

        pred_dict["Ls_v2"] = {
            "N": pred_dict["Ls_v1"]["N"] + ["dem", "shadow"], 
            "C": ["acibasi"],
            "remove_pca_cols": True
        }

        # This dataset keep all the columns, plus the PCA ones,
        # in order to assess the importance of the PCA with the spectral bands.
        pred_dict["Ls_v3"] = {
            "PCA": {
                "spring": list(cols[cols.str.endswith("spring")]),
                "summer": list(cols[cols.str.endswith("summer")])
            },
            "N": pred_dict["Ls_v2"]["N"],
            "C": pred_dict["Ls_v2"]["C"],
            "remove_pca_cols": False
        }

        self.predictor_sets = pred_dict

    def predictor_groups(self):
        """Show the predictor set keys."""
        print(self.predictors_set.keys())

    def split(self, pred_id: str, label_col: str):
        """Select Train/Test data"""
        pred_vars = self.predictor_sets[pred_id]
        # Save the key that specifies if the reduced columns should be kept
        # Remove from the rest of the dict in order to group all columns
        remove_pca_cols = pred_vars.pop('remove_pca_cols', None)

        # Get target columns and remove the NA data
        target_cols = [
            x for v in pred_vars.values()
            for x in v
            if type(v) == list
        ]

        target_pca = [
            z for v in pred_vars.values()
            if isinstance(v, dict) for x in v.values() for z in x]

        # Group the above columns and remove duplicates
        target_cols = list(set(target_cols + target_pca))
        # Create the training dataset
        X = self.matrix[target_cols + [label_col]].dropna()

        # Store predictors.
        X_pred = []
    
        # Perform the PCA reduction
        if "PCA" in pred_vars.keys():

            for sfx, pca_cols in pred_vars["PCA"].items():

                pca = Reduction(X, pca_cols, 'code_v1_reclass')
                # exp_var = pca.pca.explained_variance_ratio_
                # Discard PCs with less than 0.01 explained variance
                # selected_pcs = [str(i + 1) for i, v in enumerate(exp_var) if v > 0.01]
                selected_pca = [str(i + 1) for i in range(0, 3)]
                # Select target PCA variable names
                pca_varnames = ["PCA" + i for i in selected_pca]
                selected_pca_df = pca.df[pca_varnames]
                # Rename them to take into account the reduced variables name
                selected_pca_df.columns = ["PCA" + sfx + "_" + i for i in selected_pca]

                X_pred.append(selected_pca_df)

        if remove_pca_cols and (len(target_pca) > 0):
            # Finding duplicates
            duplicates = list(set(target_pca) & set(target_cols))
            # Remove the columns inside both lists
            target_cols = [i for i in target_cols if i not in duplicates]
            # The result is a list of columns which have not been reduced

        X_pred.append(X[target_cols])
        X_pred = pd.concat(X_pred, axis=1)

        # Split data between predictor variables and labels
        y = X.pop("code_v1_reclass") # pd.Series

        X_train, X_test, y_train, y_test = train_test_split(
            X_pred, 
            y, 
            stratify=y, 
            random_state=327, 
            test_size=0.3
        )

        return X_train, X_test, y_train, y_test


class Pipeline:

    def __init__(
            self,
            y: pd.DataFrame | pd.Series,
            under_strategy: str = None,
            over_strategy: str = None,
            X: pd.DataFrame | None = None,
            categorical_predictors: list = None
        ):
        """
        Export pipeline for model training.
        -------------------------------------

        Create the pipeline defined to train the ML models to perform
        shrub/tree classificatoin.
        
        Add the desired undersampling and oversampling strategies 
        to handle class imbalance.

        There are methods to cover scenarios with and without 
        resampling, and accounts for both continuous and categorical
        predictors.

        Parameters
        ----------
        X: The X_train partition. Mandatory to detect categorical data.
        y: Trained labels. Mandatory to perform TomekLinks.
        under_strategy: One of the names for undersampling methods availables.
        categorical_predictors : list, optional
            List of column names in the training set that represent 
            categorical predictors. If provided, SMOTENC is used instead of
            SMOTE to properly handle categorical features.

        Notes
        -----
        - Classes with more than 250 samples are considered majority classes 
        for undersampling. 
        - Undersampling techniques implemented:
            - 'random'
                Random undersampling (via a custom strategy)
            - 'tomeklinks'
                Tomek Links (removing borderline samples between classes)
            - 'none' (identity transform, no undersampling)
        - Oversampling techniques implemented:
            - adasyn
                Adaptive synthetic oversampling of minority class.
            - 'smote'
                SMOTE / SMOTENC (depending on whether categorical predictors
                are present)
            - 'none' (identity transform, no oversampling)

        - NearMiss undersampling is deprecated. A warning may appear if the
        requested number of samples exceeds the available count. In that case, 
        all available samples are returned.

        Attributes
        ----------
        under_samplers : dict
            Dictionary mapping undersampling strategy names to their corresponding objects.
        over_samplers : dict
            Dictionary mapping oversampling strategy names to their corresponding objects.
        """
        self.y = y

        # Define under sample approach
        if under_strategy == "tomeklinks":
            self.under = self.tomek()
        elif under_strategy == "random":
            self.under = self.random()
        else:
            self.under = self.none()
            # warnings.warn(f"Under strategy '{under_strategy}' is not a valid one. Set to None")

        # Define augmentation techniques
        if over_strategy == "adasyn":
            self.over = self.adasyn()
        elif over_strategy == "smote":
            if categorical_predictors is None:
                # Use standard SMOTE
                self.over = self.smote()
            else:
                # Get indices of categorical predictor columns
                col_idxs = [X.columns.get_loc(col) for col in categorical_predictors]
                # Use SMOTENC to properly handle categorical features during oversampling
                self.over = self.smotenc(col_idxs)
        else:
            self.over = self.none()
            # warnings.warn(f"Over strategy '{over_strategy}' is not a valid one. Set to None")
        
        self.under_name = under_strategy
        self.over_name = over_strategy

    def add_model(self, clf):
        """
        Create imbalance-learn pipeline
        
        Combining under sampling and data augmentation techniques.
        
        Parameters
        ----------
        clf : sklearn classifier
            Model the pipeline is created for.
        """
        self.imb_pipe = imbPipe(
            steps=[
                ("scaler", MinMaxScaler()),
                ('over', self.over),
                ('under', self.under),
                ('clf', clf)
            ])

    def grid_search(self, param_grid: dict):
        """
        Initialize and configure GridSearchCV object for the pipeline.

        Parameters
        ----------
        param_grid : dict
            Dictionary with parameters names (str) as keys and lists of 
            parameter settings to try as values.
        """

        # Use StratifiedKFold to preserve class distribution in each fold
        # Note: 2 splits are used due to sklearn2pmml compatibility issues:
        # https://github.com/jpmml/sklearn2pmml/issues/293#issuecomment-896759809
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        # Configure GridSearchCV with balanced accuracy scoring,
        # parallel computation, and error raising on failure
        grid_search = GridSearchCV(
            estimator=self.imb_pipe,
            param_grid=param_grid,
            scoring='balanced_accuracy',
            error_score='raise',
            cv=cv,
            # IMPORTANT: Avoid parallelism deadlock (n_jobs=1)
            # By default, GridSearchCV uses joblib (n_jobs=-1),
            # which can launch many parallel fits. Each pipeline holds data
            # copies (SMOTE resampled sets), which are big arrays kept in
            #  memory. After several loops, workers may deadlock or thrash
            #  memory/CPU, appearing like a “hang.”
            n_jobs=1,
            # verbose=3
        )

        return grid_search, cv

    def none(self):
        """Define the absence of strategy function."""
        return FunctionTransformer(func=lambda x: x, validate=False)

    def random(self):
        """Define random strategies."""
        return RandomUnderSampler(
            sampling_strategy=self.undersampling, random_state=42)

    def tomek(self):
        """
        TomekLinks undersampler

        Important: It only removes the tomek links, if the majority classes do
        not have any, the labels remains untouched.

        Only the labels up to 250 values will be down sampled.
        """
        # 'sampling_strategy' as a dict for cleaning methods is not supported.
        # Give a list of the classes to be targeted by the sampling.
        # Classes to be targeted:
        cls_dict = self.undersampling(self.y)
        y_res_values = [cls for cls, val in cls_dict.items() if val == 250]
        return TomekLinks(sampling_strategy=y_res_values)

    def adasyn(self):
        """
        In some partitions, this message is raisen:

            RuntimeError: Not any neigbours belong to the majority class.
            This case will induce a NaN case with a division by zero.
            ADASYN is not suited for this specific dataset. Use SMOTE instead.

            Also, when the minority amount of labels is low, the augmentation
            procedure does not work, only the percentage approach works.
        """
        return ADASYN(
            sampling_strategy=self.augmentation_perc,
            n_neighbors=2,
            random_state=42
        )
    
    def smote(self):

        return SMOTE(
            sampling_strategy=self.augmentation,
            k_neighbors=2,
            random_state=42
        )

    def smotenc(self, cf):

        return SMOTENC(
            sampling_strategy=self.augmentation,
            k_neighbors=2,
            categorical_features=cf,
            random_state=42
        )
    
    def undersampling(self, y):
        """
        Define the undersampling strategy for imbalanced classes.

        Function taking y and returns a dict. The keys correspond to the targeted classes. The values correspond to the desired number of samples for each class.

        This function generates a sampling dictionary that can be passed to 
        imbalanced-learn resampling methods (e.g., RandomUnderSampler). 
        Classes with more than 250 samples are reduced to 250, while 
        minority classes (<= 250 samples) are left unchanged.

        Parameters
        ----------
        y : array-like
            Target labels from which class distribution will be calculated.

        Returns
        -------
        sampling_dict : dict
            A dictionary mapping each class label to the desired number of samples.
            Only classes with more than 250 samples are limited.
        """

        sampling_dict = {}

        # Count the number of samples per class
        cls_counter = Counter(y)

        # Reduce large classes to a maximum of 250 samples
        for cls, count in cls_counter.items():
            if count > 250:
                sampling_dict[cls] = 250

        return sampling_dict
    
    def augmentation(self, y):
        """
        Define the data augmentation strategy for imbalanced classes.

        This function generates a sampling dictionary that can be passed to 
        imbalanced-learn resampling methods (e.g., SMOTE). Classes with less
        than 150 samples are synthetically augmented to 150, while majority
        classes (>= 150 samples) are left unchanged.

        Parameters
        ----------
        y : array-like
            Target labels from which class distribution will be calculated.

        Returns
        -------
        sampling_dict : dict
            A dictionary mapping each class label to the desired number of samples.
            Only classes with less than 150 samples are augmented.
        """
        
        sampling_dict = {}

        # Count the number of samples per class
        cls_counter = Counter(y)
        
        # Increase small classes to 150 samples
        for cls, count in cls_counter.items():
            # criteria to select labels to augment
            if count < 150:
                sampling_dict[cls] = 150
            else:
                sampling_dict[cls] = count

        return sampling_dict
    
    def augmentation_perc(self, y):
        """
        Define the data augmentation strategy for imbalanced classes.

        This function generates a sampling dictionary that can be passed to 
        imbalanced-learn resampling methods (e.g., SMOTE). Classes with less
        than 150 samples are synthetically augmented to a number which is
        the 30% of the target class samples count. Majority classes
        (>= 150 samples) are left unchanged.

        Parameters
        ----------
        y : array-like
            Target labels from which class distribution will be calculated.

        Returns
        -------
        sampling_dict : dict
            A dictionary mapping each class label to the desired number of samples.
            Only classes with less than 150 samples are augmented.
        """
        
        sampling_dict = {}

        # Count the number of samples per class
        cls_counter = Counter(y)
        
        # Increase small classes to 30% of their samples
        for cls, count in cls_counter.items():
            # criteria to select labels to augment
            if count < 150:
                # Percentage of labels to increase (30%)
                to_increase = int(round(count * 0.3, 0))
                sampling_dict[cls] = count + to_increase
            else:
                sampling_dict[cls] = count

        return sampling_dict



class Model:

    models = {
        # Random Forest
        'rf': {
            'clf': RandomForestClassifier(random_state=42, class_weight="balanced"),
            # Define the parameter grid fro GridSearchCV
            'param_grid': {
                'clf__n_estimators': [500, 600],
                'clf__min_samples_leaf': [5, 10, 15, 20],
                'clf__min_impurity_decrease': [0.0, 0.05, 0.1],
            }
        },
        # Supported Vector Machines
        'svm': {
            # Much faster than SVC and less likely to stall
            'clf': SVC(random_state=42, class_weight="balanced"),
            'param_grid': {
                'clf__C': [5, 15, 25],
                'clf__kernel': ['rbf', 'sigmoid'],
                # 'clf__kernel': ['poly', 'rbf', 'sigmoid'],
                # 'clf__degree': [1, 2, 3],
                'clf__gamma': ['scale', 'auto']
            }
        }
    }

    def __init__(self, model_key):
        """Set a model object pointing to one of the predefined models."""
        self.key = model_key
    
    def get_clf(self):
        return self.models[self.key]["clf"]

    def get_grid(self):
        return self.models[self.key]["param_grid"]

    def add_params(self, params):
        """Include the best parameters searched by GridSearchCV"""
        updated_clf = self.models[self.key]["clf"].set_params(**params)
        self.models[self.key]["clf"] = updated_clf

    def omission_error(self, cm):
        """
        Compute omission errors for each class.

        Parameters
        ----------
        cm : ndarray of shape (n_classes, n_classes)
            Confusion matrix where the i-th row represents the true class
            and the j-th column represents the predicted class.

        Returns
        -------
        list of float
            Omission error for each class, computed as the proportion of 
            misclassified samples relative to the true total of that class.
        
        Examples
        --------
        >>> import numpy as np
        >>> cm = np.array([[8, 2],
        ...                [1, 9]])
        >>> self.omission_error(cm)
        [0.2, 0.1]

        Explanation:
        - For class 0 (row 0): 2 misclassified / 10 total = 0.2
        - For class 1 (row 1): 1 misclassified / 10 total = 0.1
        """
        # Number of classes (rows of the confusion matrix)
        ncls = cm.shape[0]

        # Store omission errors for each class
        omerrs = []

        for cls in range(ncls):
            # Extract all predictions for the current true class (row)
            cls_truth = cm[cls,]

            # Misclassified samples = all predictions except the diagonal element
            # https://stackoverflow.com/a/19286855/23551600
            cls_omerrs = cls_truth[np.arange(len(cls_truth)) != cls]

            # Omission error = (misclassified samples) / (total samples of true class)
            omerrs.append(cls_omerrs.sum() / cls_truth.sum())

        return omerrs

    def commission_error(self, cm):
        """
        Compute commission errors for each class.

        Parameters
        ----------
        cm : ndarray of shape (n_classes, n_classes)
            Confusion matrix where the i-th row represents the true class
            and the j-th column represents the predicted class.

        Returns
        -------
        list of float
            Commission error for each class, computed as the proportion of 
            misclassified samples relative to the predicted total of that class.

        Examples
        --------
        >>> import numpy as np
        >>> cm = np.array([[8, 2],
        ...                [1, 9]])
        >>> self.commission_error(cm)
        [0.1111111111111111, 0.18181818181818182]

        Explanation:
        - For class 0 (column 0): 1 misclassified / 9 predicted = 0.111...
        - For class 1 (column 1): 2 misclassified / 11 predicted = 0.181...
        """
        # Number of classes (columns of the confusion matrix)
        ncls = cm.shape[1]

        # Store commission errors for each class
        coerrs = []

        for cls in range(ncls):
            # Extract all samples predicted as the current class (column)
            cls_truth = cm[:, cls]

            # Misclassified samples = all true classes except the diagonal element
            cls_coerrs = cls_truth[np.arange(len(cls_truth)) != cls]

            # Commission error = (misclassified samples) / (total predicted as this class)
            coerrs.append(cls_coerrs.sum() / cls_truth.sum())

        return coerrs
    
    def compute_metrics(self, grid_search, X_test, y_test, pipe_name, pred_id):
        """
        Compute and store evaluation metrics for a fitted GridSearchCV model.

        Parameters
        ----------
        grid_search : GridSearchCV
            The fitted GridSearchCV object containing the best estimator.
        X_test, y_test: pd.DataFrame, pd.Series
            Data to test the model.
        pipe_name : str
            Name of the pipeline associated with the model.
        pred_id : str
            Identifier for the used predictor variables.

        Returns
        -------
        tuple
            A tuple containing:
            - cm_dict (dict): Confusion matrix dictionary.
            - metrics (dict): Model error metrics.
        """
        stats_cols = ['model', 'metric', 'label_code', 'data', 'pipe_name', 'pred_id']
        # Generate predictions on the test set
        y_predict = grid_search.predict(X_test)

        # Compute global performance metrics
        accuracy = accuracy_score(y_test, y_predict)
        balanced_accuracy = balanced_accuracy_score(y_test, y_predict)
        kappa = cohen_kappa_score(y_test, y_predict)

        # Extract unique predicted labels (to align confusion matrix properly)
        y_labels = np.unique(y_predict)

        # Build the confusion matrix using only predicted labels
        cm = confusion_matrix(y_test, y_predict, labels=y_labels)

        # Class-specific omission and commission errors
        om_err = self.omission_error(cm)
        co_err = self.commission_error(cm)

        # Producer’s accuracy (1 - omission error) per class
        prod_acc = [1 - oe for oe in om_err]

        # User’s accuracy (1 - commission error) per class
        user_acc = [1 - ce for ce in co_err]

        # Store global metrics
        metrics = []
        metrics.append([self.key, 'overall_accuracy', None, accuracy, pipe_name, pred_id])
        metrics.append([self.key, 'balanced_accuracy', None, balanced_accuracy, pipe_name, pred_id])
        metrics.append([self.key, 'kappa', None, kappa, pipe_name, pred_id])

        # Store per-class producer’s accuracy
        for i, label in enumerate(y_labels):
            metrics.append([self.key, 'producer_acc', label, prod_acc[i], pipe_name, pred_id])

        # Store per-class user’s accuracy
        for i, label in enumerate(y_labels):
            metrics.append([self.key, 'user_acc', label, user_acc[i], pipe_name, pred_id])

        # Prepare Confusion Matrix dictionary
        cm_keys = ['cm', 'labels', 'model', 'pipe', 'pred_id']
        cm_vals = [cm, y_labels, self.key, pipe_name, pred_id]
        cm_dict = dict(zip(cm_keys, cm_vals))
        return cm_dict, pd.DataFrame(metrics, columns=stats_cols)


class Reduction:
    def __init__(self, df: pd.DataFrame, predictors: list, label_col: str):
        """
        Perform Principal Component Analysis (PCA) on selected predictors.

        Note: The target dataframe mustn't have NA data.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing predictor variables and labels.
        predictors : list
            List of column names in `df` to use as predictors for PCA.
        label_col : str
            Column name in `df` containing the labels. These labels are kept
            in the PCA dataframe to allow visualization and grouping.

        Attributes
        ----------
        df : pandas.DataFrame
            DataFrame containing the labels and the PCA-transformed components.
        pca_cols : list of str
            Names of the PCA component columns (e.g., ["PCA1", "PCA2", ...]).
        pca : sklearn.decomposition.PCA
            Fitted PCA object.
        label_code : str
            Name of the label column used.
        loads : pandas.DataFrame
            Loadings of each predictor variable on the principal components,
            with predictors as rows and components as columns.
        """

        # Standardize predictors (mean=0, variance=1) before PCA
        X = StandardScaler().fit_transform(df[predictors])

        # Fit PCA with 5 components
        pca = PCA(n_components=5, random_state=42)
        pca.fit(X)

        # Apply PCA transformation to predictors (NumPy array)
        transformation_np = pca.transform(X)

        # Create readable names for PCA components
        pca_cols = ['PCA%i' % i for i in range(1, pca.n_components_ + 1)]

        # Convert PCA results into a DataFrame with proper column names
        transformation_df = pd.DataFrame(transformation_np, columns=pca_cols,
                                         index=df.index)

        # Combine PCA components with labels for easier analysis/plotting
        df_pca = pd.concat([df[label_col], transformation_df], axis=1)

        # Store as class attributes
        self.df = df_pca
        self.pca_cols = pca_cols
        self.pca = pca
        self.label_code = label_col

        # Compute variable loadings (importance of predictors in each component)
        self.loads = pd.DataFrame(pca.components_.T, columns=pca_cols, index=predictors)
    
    def explained_variance(self):
        """
        Plot the explained variance ratio of each principal component.

        Notes
        -----
        Shows the proportion of variance explained by each PCA component
        as a bar plot.
        """
        plt.bar(self.pca_cols, self.pca.explained_variance_ratio_, color='gold')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(self.pca_cols)

    def plot_loadings(self):
        """
        Plot the loadings of each predictor variable on the principal components.

        Notes
        -----
        Creates horizontal bar plots, one for each principal component,
        showing how strongly each predictor contributes to it.
        """
        fig, axs = plt.subplots(self.loads.shape[1], figsize=(5, 15), sharex=True)

        for i, pca_col in enumerate(self.pca_cols):
            self.loads.iloc[:, i].plot.barh(ax=axs[i])
            axs[i].set_title(pca_col)

class CMatrix:
    def __init__(self, cm):
        """
        Plot a pretty confusion matrix with seaborn
        Created on Mon Jun 25 14:17:37 2018
        @author: Wagner Cipriano - wagnerbhbr - gmail - CEFETMG / MMC
        https://github.com/wcipriano/pretty-print-confusion-matrix
        REFerences:
          https://www.mathworks.com/help/nnet/ref/plotconfusion.html
          https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
          https://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python
          https://www.programcreek.com/python/example/96197/seaborn.heatmap
          https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/31720054
          http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

        :cm: dataframe (pandas) without totals
        """
        self.cm = cm

    def get_new_fig(self, fn, figsize=[9, 9]):
        """Init graphics"""
        # plt.rcParams['text.usetex'] = True
        fig1 = plt.figure(fn, figsize)
        ax1 = fig1.gca()  # Get Current Axis
        ax1.cla()  # clear existing plot
        return fig1, ax1


    def configcell_text_and_colors(
        self, array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0
    ):
        """
        config cell text and colors
        and return text elements to add and to dell
        @TODO: use fmt
        """
        text_add = []
        text_del = []
        cell_val = array_df[lin][col]
        tot_all = array_df[-1][-1]
        per = (float(cell_val) / tot_all) * 100
        curr_column = array_df[:, col]
        ccl = len(curr_column)

        # last line  and/or last column
        if (col == (ccl - 1)) or (lin == (ccl - 1)):
            # tots and percents
            if cell_val != 0:
                if (col == ccl - 1) and (lin == ccl - 1):
                    tot_rig = 0
                    for i in range(array_df.shape[0] - 1):
                        tot_rig += array_df[i][i]
                    per_ok = (float(tot_rig) / cell_val) * 100
                elif col == ccl - 1:
                    tot_rig = array_df[lin][lin]
                    per_ok = (float(tot_rig) / cell_val) * 100
                elif lin == ccl - 1:
                    tot_rig = array_df[col][col]
                    per_ok = (float(tot_rig) / cell_val) * 100
                per_err = 100 - per_ok
            else:
                per_ok = per_err = 0

            per_ok_s = ["%.1f%%" % (per_ok), "100%"][int(per_ok == 100)]

            # text to DEL
            text_del.append(oText)

            # text to ADD
            font_prop = fm.FontProperties(weight="bold", size=fz)
            text_kwargs = dict(
                color="k",
                ha="center",
                va="center",
                gid="sum",
                fontproperties=font_prop,
            )
            # Check if the column is the total column + total row
            # Transform color to white (black made poor visualization)
            if col == lin:
                text_kwargs['color'] = "w"
            lis_txt = ["%d" % (cell_val), per_ok_s, "%.1f%%" % (per_err)]
            lis_kwa = [text_kwargs]
            dic = text_kwargs.copy()
            dic["color"] = "g"
            lis_kwa.append(dic)
            dic = text_kwargs.copy()
            dic["color"] = "r"
            lis_kwa.append(dic)
            lis_pos = [
                (oText._x, oText._y - 0.3),
                (oText._x, oText._y),
                (oText._x, oText._y + 0.3),
            ]
            for i in range(len(lis_txt)):
                newText = dict(
                    x=lis_pos[i][0],
                    y=lis_pos[i][1],
                    text=lis_txt[i],
                    kw=lis_kwa[i],
                )
                text_add.append(newText)

            # set background color for sum cells (last line and last column)
            carr = [0.27, 0.30, 0.27, 1.0]
            if (col == ccl - 1) and (lin == ccl - 1):
                carr = [0.17, 0.20, 0.17, 1.0]
            facecolors[posi] = carr

        else:
            if per > 0:
                txt = "%s\n%.1f%%" % (cell_val, per)
            else:
                if show_null_values == 0:
                    txt = ""
                elif show_null_values == 1:
                    txt = "0"
                else:
                    txt = "0\n0.0%"
            oText.set_text(txt)

            # main diagonal
            if col == lin:
                # set color of the textin the diagonal to black
                oText.set_color("k")
                # set background color in the diagonal to blue
                facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
            else:
                oText.set_color("r")

        return text_add, text_del


    def insert_totals(self, df_cm):
        """insert total column and line (the last ones)"""
        sum_col = []
        for c in df_cm.columns:
            sum_col.append(df_cm[c].sum())
        sum_lin = []
        for item_line in df_cm.iterrows():
            sum_lin.append(item_line[1].sum())
        df_cm["sum_lin"] = sum_lin
        sum_col.append(np.sum(sum_lin))
        df_cm.loc["sum_col"] = sum_col

    def pp_matrix(self,
        annot=True,
        cmap="Oranges",
        fmt=".2f",
        fz=11,
        lw=0.5,
        cbar=False,
        figsize=[8, 8],
        show_null_values=0,
        pred_val_axis="y",
        cmtitle = "Confusion matrix",
        savefig = ""):
        """
        Print confusion matrix.

        params:
          annot          print text in each cell
          cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
          fz             fontsize
          lw             linewidth
          pred_val_axis  where to show the prediction values (x or y axis)
                          'col' or 'x': show predicted values in columns (x axis) instead lines
                          'lin' or 'y': show predicted values in lines   (y axis)
        """
        if pred_val_axis in ("col", "x"):
            xlbl = "Predicted"
            ylbl = "Actual"
        else:
            xlbl = "Actual"
            ylbl = "Predicted"
            df_cm = df_cm.T

        # create "Total" column
        self.insert_totals(self.cm)

        # this is for print allways in the same window
        fig, ax1 = self.get_new_fig("Conf matrix default", figsize)

        ax = sn.heatmap(
            self.cm,
            annot=annot,
            annot_kws={"size": fz},
            linewidths=lw,
            ax=ax1,
            cbar=cbar,
            cmap=cmap,
            linecolor="k",
            fmt=fmt
        )
        # re-enable outer spines
        # https://stackoverflow.com/a/71574828/23551600
        sn.despine(left=False, right=False, top=False, bottom=False)
        # Change column and line totals names
        xlabels = ax.get_xticklabels()
        ylabels = ax.get_yticklabels()
        for i, (xlab, ylab) in enumerate(zip(xlabels, ylabels)):
            if xlab.get_text().startswith('sum') or ylab.get_text().startswith('sum'):
                x_newtext = f"Sum {xlab.get_text().split('_')[1]}."
                y_newtext = f"Sum {ylab.get_text().split('_')[1]}."

                xlabels[i].set_text(x_newtext)
                ylabels[i].set_text(y_newtext)

        # set ticklabels rotation
        ax.set_xticklabels(xlabels, rotation=45, fontsize=10,ha="right", rotation_mode="anchor")
        ax.set_yticklabels(ylabels, rotation=0, fontsize=10)

        # Turn off all the ticks
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        # face colors list
        quadmesh = ax.findobj(QuadMesh)[0]
        facecolors = quadmesh.get_facecolors()

        # iter in text elements
        array_df = np.array(self.cm.to_records(index=False).tolist())
        text_add = []
        text_del = []
        posi = -1  # from left to right, bottom to top.
        for t in ax.collections[0].axes.texts:  # ax.texts:
            pos = np.array(t.get_position()) - [0.5, 0.5]
            lin = int(pos[1])
            col = int(pos[0])
            posi += 1

            # set text
            txt_res = self.configcell_text_and_colors(
                array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values
            )

            text_add.extend(txt_res[0])
            text_del.extend(txt_res[1])

        # remove the old ones
        for item in text_del:
            item.remove()
        # append the new ones
        for item in text_add:
            ax.text(item["x"], item["y"], item["text"], **item["kw"])

        # titles and legends
        ax.set_title(cmtitle)
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        plt.tight_layout()  # set layout slim
        plt.show()
        if len(savefig) > 0:
            fig.savefig(savefig)

def loop_training(dataset: Dataset, pred_id, cm_list, stats_list, grid_list):

    X_train, X_test, y_train, y_test = dataset.split(pred_id, "ESPE_rc")

    pred_vars = dataset.predictor_sets[pred_id]
    
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

    return(cm_list, stats_list, grid_list)