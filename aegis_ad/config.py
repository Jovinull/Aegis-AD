"""Centralised configuration for the Aegis-AD pipeline.

A single immutable dataclass governs all pipeline behaviour. Hyperparameter search
spaces are kept here rather than dispersed across modules so that experimental
sweeps can be reproduced from a single configuration object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AegisConfig:
    """Pipeline-wide configuration.

    Attributes
    ----------
    data_dir, cross_sectional_file, longitudinal_file
        Locations of the OASIS-1 and OASIS-2 CSV exports.
    target_mode
        "binary"  -> y = 1 if CDR >= binary_threshold (typical clinical cut-point).
        "multiclass" -> y in {0, 0.5, 1, 2} mapped to {0, 1, 2, 3}.
    n_splits
        Number of folds for the *outer* group-stratified CV. Inner tuning CV is
        derived from this with one fewer fold.
    use_smote
        Whether to oversample the minority class with SMOTE within each training
        fold (applied strictly inside the imbalanced-learn pipeline so that
        synthetic samples never leak into validation).
    n_optuna_trials, optuna_timeout
        Optuna search budget per base learner. ``timeout`` is in seconds.
    tabnet_*
        Training schedule for the deep tabular branch.
    shap_max_samples
        Background-set cap for SHAP explanations (controls compute cost).
    """

    # -- data ----------------------------------------------------------------
    data_dir: Path = Path("dataset")
    cross_sectional_file: str = "oasis_cross-sectional.csv"
    longitudinal_file: str = "oasis_longitudinal.csv"
    output_dir: Path = Path("artifacts")

    # -- task ----------------------------------------------------------------
    target_mode: str = "binary"
    binary_threshold: float = 0.5

    # -- cross-validation ----------------------------------------------------
    n_splits: int = 5
    inner_n_splits: int = 4
    random_state: int = 42

    # -- imbalance handling --------------------------------------------------
    use_smote: bool = True
    smote_k_neighbors: int = 3

    # -- hyperparameter search ----------------------------------------------
    n_optuna_trials: int = 40
    optuna_timeout: int = 600

    # -- TabNet --------------------------------------------------------------
    tabnet_max_epochs: int = 200
    tabnet_patience: int = 25
    tabnet_batch_size: int = 256
    tabnet_virtual_batch_size: int = 64

    # -- explainability ------------------------------------------------------
    shap_max_samples: int = 200
    shap_local_examples: int = 6

    @property
    def cross_sectional_path(self) -> Path:
        return self.data_dir / self.cross_sectional_file

    @property
    def longitudinal_path(self) -> Path:
        return self.data_dir / self.longitudinal_file
