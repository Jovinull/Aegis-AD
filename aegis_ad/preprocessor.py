"""Leak-free preprocessing for Aegis-AD.

The preprocessor is a scikit-learn ``ColumnTransformer`` that lives *inside* the
imbalanced-learn pipeline executed during cross-validation. Because every fitted
component (``KNNImputer``, ``RobustScaler``, ``OneHotEncoder``) is refit per
training fold, no statistic computed on validation data ever influences the
learned transform — eliminating the classical leakage failure mode.

Imputation choice
-----------------
We use ``KNNImputer`` with ``n_neighbors=5`` rather than column-wise mean
imputation: OASIS contains conditional missingness (``ses`` and ``educ`` are
missing for the same subset of younger subjects), and KNN preserves the joint
distribution of demographic blocks substantially better than marginal mean
imputation.

Scaling choice
--------------
``RobustScaler`` (median / IQR) is preferred over ``StandardScaler`` because
volumetric measures (``etiv``, ``nwbv``) and longitudinal slopes have heavy
tails that would compromise z-score statistics, and because TabNet and
gradient boosters benefit from outlier-robust scaling at the input layer.
"""

from __future__ import annotations

from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    knn_neighbors: int = 5,
) -> Tuple[ColumnTransformer, List[str]]:
    """Return a fitted-per-fold ColumnTransformer and its post-transform names.

    Parameters
    ----------
    numeric_features, categorical_features
        Column names belonging to each block. The numeric block is imputed with
        KNN and scaled with RobustScaler; the categorical block is one-hot
        encoded with ``handle_unknown="ignore"`` so that rare ``hand="L"`` rows
        in held-out folds never break inference.
    knn_neighbors
        ``KNNImputer.n_neighbors``. The OASIS sample is small; we default to 5.

    Returns
    -------
    transformer, output_feature_names
        The transformer and the resolved output feature names (useful for SHAP).
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=knn_neighbors, weights="distance")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="if_binary"
                ),
            ),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Output names cannot be resolved until fit; we return the input-side names
    # and compute the resolved names after the first .fit() call.
    return transformer, list(numeric_features) + list(categorical_features)


def resolve_feature_names(transformer: ColumnTransformer) -> List[str]:
    """Resolve post-transform feature names from a fitted ColumnTransformer."""
    try:
        return list(transformer.get_feature_names_out())
    except Exception:  # pragma: no cover — defensive
        names: List[str] = []
        for name, trans, cols in transformer.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "get_feature_names_out"):
                names.extend(trans.get_feature_names_out(cols))
            else:
                names.extend(cols)
        return names
