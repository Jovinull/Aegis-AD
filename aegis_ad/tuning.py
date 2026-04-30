"""Bayesian hyperparameter optimisation with Optuna.

Each base learner has its own Optuna ``Study``. Tuning is run on the *training
partition only*, with an inner ``StratifiedGroupKFold`` so that subjects never
straddle inner train/validation folds. The optimisation target is the mean
ROC-AUC across the inner folds, which is a strictly proper scoring rule with
respect to the discriminative quantity of clinical interest (probability of
dementia given the features).

The final returned ``params`` dictionary uses the same key prefixes consumed by
``AegisEnsemble`` (``xgb_*``, ``lgbm_*``, ``cat_*``, ``tabnet_*``, ``meta_*``),
so the tuner output can be passed directly into ``AegisEnsemble.build``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .config import AegisConfig

# Silence Optuna's INFO chatter; warnings/errors still propagate.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# Inner CV scoring utility                                                    #
# --------------------------------------------------------------------------- #
def _cv_auc(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> float:
    """Group-aware inner CV mean ROC-AUC."""
    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    aucs = []
    for tr, va in cv.split(X, y, groups):
        est = clone(estimator)
        est.fit(X[tr], y[tr])
        proba = est.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(y[va], proba))
    return float(np.mean(aucs))


# --------------------------------------------------------------------------- #
# Per-branch search spaces                                                     #
# --------------------------------------------------------------------------- #
def _suggest_xgb(trial: optuna.Trial, scale_pos_weight: float) -> Dict[str, Any]:
    from xgboost import XGBClassifier

    params = {
        "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 200, 900, step=100),
        "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
        "xgb_lr": trial.suggest_float("xgb_lr", 1e-3, 2e-1, log=True),
        "xgb_subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "xgb_colsample": trial.suggest_float("xgb_colsample", 0.6, 1.0),
        "xgb_reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-3, 5.0, log=True),
        "xgb_reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-3, 5.0, log=True),
    }
    estimator = XGBClassifier(
        n_estimators=params["xgb_n_estimators"],
        max_depth=params["xgb_max_depth"],
        learning_rate=params["xgb_lr"],
        subsample=params["xgb_subsample"],
        colsample_bytree=params["xgb_colsample"],
        reg_alpha=params["xgb_reg_alpha"],
        reg_lambda=params["xgb_reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
        random_state=0,
    )
    return params, estimator  # type: ignore[return-value]


def _suggest_lgbm(trial: optuna.Trial, scale_pos_weight: float) -> Dict[str, Any]:
    from lightgbm import LGBMClassifier

    params = {
        "lgbm_n_estimators": trial.suggest_int("lgbm_n_estimators", 200, 900, step=100),
        "lgbm_num_leaves": trial.suggest_int("lgbm_num_leaves", 15, 127),
        "lgbm_lr": trial.suggest_float("lgbm_lr", 1e-3, 2e-1, log=True),
        "lgbm_min_child": trial.suggest_int("lgbm_min_child", 4, 30),
        "lgbm_subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
        "lgbm_colsample": trial.suggest_float("lgbm_colsample", 0.6, 1.0),
        "lgbm_reg_alpha": trial.suggest_float("lgbm_reg_alpha", 1e-3, 5.0, log=True),
        "lgbm_reg_lambda": trial.suggest_float("lgbm_reg_lambda", 1e-3, 5.0, log=True),
    }
    estimator = LGBMClassifier(
        n_estimators=params["lgbm_n_estimators"],
        num_leaves=params["lgbm_num_leaves"],
        learning_rate=params["lgbm_lr"],
        min_child_samples=params["lgbm_min_child"],
        subsample=params["lgbm_subsample"],
        colsample_bytree=params["lgbm_colsample"],
        reg_alpha=params["lgbm_reg_alpha"],
        reg_lambda=params["lgbm_reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        verbosity=-1,
        random_state=0,
    )
    return params, estimator  # type: ignore[return-value]


def _suggest_catboost(trial: optuna.Trial, scale_pos_weight: float) -> Dict[str, Any]:
    from catboost import CatBoostClassifier

    params = {
        "cat_iterations": trial.suggest_int("cat_iterations", 200, 900, step=100),
        "cat_depth": trial.suggest_int("cat_depth", 3, 8),
        "cat_lr": trial.suggest_float("cat_lr", 1e-3, 2e-1, log=True),
        "cat_l2": trial.suggest_float("cat_l2", 1e-1, 10.0, log=True),
    }
    estimator = CatBoostClassifier(
        iterations=params["cat_iterations"],
        depth=params["cat_depth"],
        learning_rate=params["cat_lr"],
        l2_leaf_reg=params["cat_l2"],
        scale_pos_weight=scale_pos_weight,
        random_state=0,
        verbose=False,
    )
    return params, estimator  # type: ignore[return-value]


_SUGGEST_REGISTRY = {
    "xgb": _suggest_xgb,
    "lgbm": _suggest_lgbm,
    "catboost": _suggest_catboost,
}


# --------------------------------------------------------------------------- #
# Public tuner                                                                 #
# --------------------------------------------------------------------------- #
class OptunaTuner:
    """Tunes each tree-based base learner independently and returns a merged
    parameter dictionary suitable for ``AegisEnsemble.build``.

    The deep-tabular branch (TabNet) is *not* tuned per fold by default — its
    optimisation budget would dominate the wall-clock cost on a small cohort,
    and TabNet's published default architecture is already a strong baseline
    (Arik & Pfister, 2021). Users who need full tuning can extend
    ``self.branches`` with a TabNet-specific sampler.
    """

    def __init__(self, cfg: AegisConfig, branches: Optional[list] = None) -> None:
        self.cfg = cfg
        self.branches = branches or list(_SUGGEST_REGISTRY.keys())

    def tune(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, scale_pos_weight: float
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        per_branch_trials = max(self.cfg.n_optuna_trials // len(self.branches), 8)

        for branch in self.branches:
            sampler = TPESampler(seed=self.cfg.random_state)
            study = optuna.create_study(direction="maximize", sampler=sampler)

            def _objective(trial: optuna.Trial) -> float:
                params, estimator = _SUGGEST_REGISTRY[branch](trial, scale_pos_weight)
                return _cv_auc(
                    estimator,
                    X,
                    y,
                    groups,
                    n_splits=self.cfg.inner_n_splits,
                    random_state=self.cfg.random_state,
                )

            study.optimize(
                _objective,
                n_trials=per_branch_trials,
                timeout=self.cfg.optuna_timeout // len(self.branches),
                show_progress_bar=False,
            )
            merged.update(study.best_params)

        return merged
