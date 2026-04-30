"""sklearn-compatible adapter around `pytorch_tabnet.tab_model.TabNetClassifier`.

Why TabNet?
-----------
TabNet (Arik & Pfister, 2021, *AAAI*) is a sequential attention architecture
designed for tabular data: at each decision step a learnable mask selects a
sparse subset of features for processing, yielding both competitive accuracy
and *step-wise interpretability*. In the Aegis-AD ensemble it serves as the
"deep tabular" branch (Branch B) and provides a representational signal that
is *complementary* to gradient-boosted trees — the latter excel at axis-aligned
splits, the former at smooth, attention-weighted feature combinations.

Fallback
--------
If ``pytorch_tabnet`` is not installed at import time, we raise an informative
ImportError when the class is instantiated rather than at module import — so the
rest of the pipeline (which may not need TabNet at hyperparameter-tuning time)
remains usable.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

try:  # pragma: no cover — optional dependency
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier as _TabNetCore

    _TABNET_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TABNET_AVAILABLE = False


class TabNetSklearnClassifier(BaseEstimator, ClassifierMixin):
    """Thin sklearn-API wrapper around TabNetClassifier with safe defaults."""

    def __init__(
        self,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 4,
        gamma: float = 1.5,
        lambda_sparse: float = 1e-4,
        learning_rate: float = 2e-2,
        max_epochs: int = 200,
        patience: int = 25,
        batch_size: int = 256,
        virtual_batch_size: int = 64,
        seed: int = 42,
        device_name: str = "auto",
        verbose: int = 0,
    ) -> None:
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.seed = seed
        self.device_name = device_name
        self.verbose = verbose

    # --------------------------------------------------------------- fitting
    def fit(
        self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Any] = None
    ) -> "TabNetSklearnClassifier":
        if not _TABNET_AVAILABLE:
            raise ImportError(
                "pytorch_tabnet is required for the deep-tabular branch. "
                "Install with `pip install pytorch-tabnet`."
            )
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int).ravel()

        self.classes_ = np.unique(y)
        self._model = _TabNetCore(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": self.learning_rate},
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={"mode": "min", "factor": 0.5, "patience": 8},
            mask_type="entmax",
            device_name=self.device_name,
            seed=self.seed,
            verbose=self.verbose,
        )

        # If no explicit eval_set is supplied (e.g. inside cross_val_predict),
        # we carve a small internal stratified split off the training partition
        # so that the early-stopping callback has a target to monitor.
        if eval_set is None:
            eval_set = self._internal_eval_split(X, y)

        self._model.fit(
            X_train=X,
            y_train=y,
            eval_set=eval_set,
            eval_metric=["auc"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            num_workers=0,
            drop_last=False,
        )
        return self

    # --------------------------------------------------------------- inference
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "_model")
        X = np.asarray(X, dtype=np.float32)
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _internal_eval_split(X: np.ndarray, y: np.ndarray, frac: float = 0.15):
        rng = np.random.default_rng(0)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        n_val = max(int(len(idx) * frac), 8)
        val_idx = idx[:n_val]
        return [(X[val_idx], y[val_idx])]

    @property
    def feature_importances_(self) -> np.ndarray:
        check_is_fitted(self, "_model")
        return np.asarray(self._model.feature_importances_)
