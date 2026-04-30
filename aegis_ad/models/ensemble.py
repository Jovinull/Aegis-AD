"""The Aegis heterogeneous stacking ensemble.

Two design points distinguish this implementation from the stock
``sklearn.ensemble.StackingClassifier``:

1. **Group-aware out-of-fold meta features.** ``StackingClassifier.fit`` does
   not forward ``groups`` to its internal cross-validation, which would silently
   leak subject identity across folds in the longitudinal cohort. We instead
   build a ``GroupAwareStackingClassifier`` that performs OOF prediction with a
   group-respecting splitter (StratifiedGroupKFold).

2. **Heterogeneous branches.** Branch A bundles three gradient boosters
   (XGBoost, LightGBM, CatBoost) — each captures slightly different aspects of
   the loss surface (XGBoost: histogram + L1/L2; LightGBM: leaf-wise growth;
   CatBoost: ordered boosting + symmetric trees). Branch B is a TabNet
   classifier providing attention-weighted dense representations. The meta-
   learner is an L2-regularised logistic regression — the simplest combiner
   that controls the effective ensemble VC-dimension on a small (∼800-row)
   cohort, in line with the Stacked Generalization principle of Wolpert (1992).

References
----------
Wolpert (1992) *Stacked Generalization*. Neural Networks 5(2).
Arik & Pfister (2021) *TabNet: Attentive Interpretable Tabular Learning*. AAAI.
Diogo et al. (2022) *Early diagnosis of Alzheimer's disease using machine
  learning: a multi-diagnostic, generalizable approach*. Alzheimer's Research &
  Therapy 14, 107.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.validation import check_is_fitted

# Optional / heavy dependencies are imported lazily inside the factories so that
# unit tests on machines without GPU or CatBoost still succeed.
try:
    from xgboost import XGBClassifier

    _XGB_OK = True
except Exception:  # noqa: BLE001
    _XGB_OK = False

try:
    from lightgbm import LGBMClassifier

    _LGBM_OK = True
except Exception:  # noqa: BLE001
    _LGBM_OK = False

try:
    from catboost import CatBoostClassifier

    _CAT_OK = True
except Exception:  # noqa: BLE001
    _CAT_OK = False

from .tabnet_wrapper import TabNetSklearnClassifier


# ============================================================================ #
# Group-aware stacking
# ============================================================================ #
class GroupAwareStackingClassifier(BaseEstimator, ClassifierMixin):
    """Stacking classifier that respects subject grouping during OOF generation.

    Parameters
    ----------
    base_estimators
        Sequence of ``(name, estimator)`` tuples. Each estimator must implement
        ``fit`` and ``predict_proba``.
    final_estimator
        Meta-learner trained on the OOF predictions of the base estimators.
    cv
        A *group-aware* splitter (e.g. ``StratifiedGroupKFold``).
    passthrough
        If True, raw features are concatenated to the OOF predictions before
        feeding the meta-learner. Off by default — for an 800-row tabular
        cohort, passthrough tends to inflate meta-learner variance.
    """

    def __init__(
        self,
        base_estimators: Sequence[Tuple[str, BaseEstimator]],
        final_estimator: BaseEstimator,
        cv: Any,
        passthrough: bool = False,
    ) -> None:
        self.base_estimators = list(base_estimators)
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough

    # ----------------------------------------------------------------- fit
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> "GroupAwareStackingClassifier":
        X = np.asarray(X)
        y = np.asarray(y).astype(int).ravel()

        self.classes_ = np.unique(y)
        n_classes = self.classes_.size
        n_meta_cols = (n_classes - 1) if n_classes == 2 else n_classes

        n_samples = X.shape[0]
        n_base = len(self.base_estimators)
        meta_features = np.zeros((n_samples, n_base * n_meta_cols))

        # ---- 1. OOF meta-feature generation, respecting groups ------------
        for j, (name, est) in enumerate(self.base_estimators):
            for tr_idx, va_idx in self.cv.split(X, y, groups):
                est_fold = clone(est)
                est_fold.fit(X[tr_idx], y[tr_idx])
                proba = est_fold.predict_proba(X[va_idx])
                if n_classes == 2:
                    meta_features[va_idx, j] = proba[:, 1]
                else:
                    meta_features[va_idx, j * n_classes : (j + 1) * n_classes] = proba

        # ---- 2. Refit each base on the full training partition ------------
        self.fitted_base_: List[Tuple[str, BaseEstimator]] = []
        for name, est in self.base_estimators:
            fitted = clone(est).fit(X, y)
            self.fitted_base_.append((name, fitted))

        # ---- 3. Train meta-learner on OOF predictions ---------------------
        meta_X = self._compose_meta_input(meta_features, X)
        self.final_estimator_ = clone(self.final_estimator).fit(meta_X, y)
        self._n_meta_cols = n_meta_cols
        return self

    # ----------------------------------------------------------------- predict
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "final_estimator_")
        X = np.asarray(X)
        n_classes = self.classes_.size
        cols = []
        for _, est in self.fitted_base_:
            proba = est.predict_proba(X)
            if n_classes == 2:
                cols.append(proba[:, 1:2])
            else:
                cols.append(proba)
        meta_features = np.hstack(cols)
        meta_X = self._compose_meta_input(meta_features, X)
        return self.final_estimator_.predict_proba(meta_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    # ----------------------------------------------------------------- helpers
    def _compose_meta_input(self, meta: np.ndarray, X: np.ndarray) -> np.ndarray:
        if self.passthrough:
            return np.hstack([meta, X])
        return meta


# ============================================================================ #
# AegisEnsemble factory
# ============================================================================ #
class AegisEnsemble:
    """Factory that assembles the canonical Aegis-AD stacking ensemble.

    Usage
    -----
    >>> ensemble = AegisEnsemble(random_state=42).build(
    ...     n_splits=4, scale_pos_weight=2.3, params=tuned_params)
    >>> ensemble.fit(X_train, y_train, groups=g_train)
    >>> ensemble.predict_proba(X_test)
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    # --------------------------------------------------------------- factories
    def _xgb(self, params: Dict[str, Any], scale_pos_weight: float) -> BaseEstimator:
        if not _XGB_OK:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(
            n_estimators=params.get("xgb_n_estimators", 600),
            max_depth=params.get("xgb_max_depth", 4),
            learning_rate=params.get("xgb_lr", 0.05),
            subsample=params.get("xgb_subsample", 0.85),
            colsample_bytree=params.get("xgb_colsample", 0.85),
            reg_alpha=params.get("xgb_reg_alpha", 0.1),
            reg_lambda=params.get("xgb_reg_lambda", 1.0),
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )

    def _lgbm(self, params: Dict[str, Any], scale_pos_weight: float) -> BaseEstimator:
        if not _LGBM_OK:
            raise ImportError("lightgbm is not installed.")
        return LGBMClassifier(
            n_estimators=params.get("lgbm_n_estimators", 600),
            num_leaves=params.get("lgbm_num_leaves", 31),
            learning_rate=params.get("lgbm_lr", 0.05),
            min_child_samples=params.get("lgbm_min_child", 8),
            subsample=params.get("lgbm_subsample", 0.85),
            colsample_bytree=params.get("lgbm_colsample", 0.85),
            reg_alpha=params.get("lgbm_reg_alpha", 0.1),
            reg_lambda=params.get("lgbm_reg_lambda", 1.0),
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    def _catboost(
        self, params: Dict[str, Any], scale_pos_weight: float
    ) -> BaseEstimator:
        if not _CAT_OK:
            raise ImportError("catboost is not installed.")
        return CatBoostClassifier(
            iterations=params.get("cat_iterations", 600),
            depth=params.get("cat_depth", 6),
            learning_rate=params.get("cat_lr", 0.05),
            l2_leaf_reg=params.get("cat_l2", 3.0),
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            verbose=False,
        )

    def _tabnet(self, params: Dict[str, Any]) -> BaseEstimator:
        return TabNetSklearnClassifier(
            n_d=params.get("tabnet_nd", 16),
            n_a=params.get("tabnet_na", 16),
            n_steps=params.get("tabnet_steps", 4),
            gamma=params.get("tabnet_gamma", 1.5),
            lambda_sparse=params.get("tabnet_lambda_sparse", 1e-4),
            learning_rate=params.get("tabnet_lr", 2e-2),
            max_epochs=params.get("tabnet_max_epochs", 200),
            patience=params.get("tabnet_patience", 25),
            batch_size=params.get("tabnet_batch_size", 256),
            virtual_batch_size=params.get("tabnet_vbs", 64),
            seed=self.random_state,
        )

    # --------------------------------------------------------------- build
    def build(
        self,
        n_splits: int = 4,
        scale_pos_weight: float = 1.0,
        params: Optional[Dict[str, Any]] = None,
        include_tabnet: bool = True,
        passthrough: bool = False,
    ) -> GroupAwareStackingClassifier:
        params = params or {}
        bases: List[Tuple[str, BaseEstimator]] = []
        if _XGB_OK:
            bases.append(("xgb", self._xgb(params, scale_pos_weight)))
        if _LGBM_OK:
            bases.append(("lgbm", self._lgbm(params, scale_pos_weight)))
        if _CAT_OK:
            bases.append(("catboost", self._catboost(params, scale_pos_weight)))
        if include_tabnet:
            bases.append(("tabnet", self._tabnet(params)))

        if not bases:
            raise RuntimeError(
                "No base learners are available — install at least "
                "one of xgboost, lightgbm, catboost."
            )

        meta = LogisticRegression(
            penalty="l2",
            C=params.get("meta_C", 1.0),
            solver="lbfgs",
            max_iter=2000,
            random_state=self.random_state,
        )

        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        return GroupAwareStackingClassifier(
            base_estimators=bases,
            final_estimator=meta,
            cv=cv,
            passthrough=passthrough,
        )
