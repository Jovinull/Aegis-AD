"""Temporal and demographic feature engineering for Aegis-AD.

Strategy
--------
For *longitudinal* subjects (OAS2_*) we collapse all visits per subject into a
single observation enriched with longitudinal *dynamics*:

    Δ-features      : last-minus-baseline of nwbv, mmse, etiv, asf, cdr.
    slope-features  : OLS slope of each marker against ``mr_delay`` (days).
    visit-features  : visit count, total follow-up duration, max-CDR.

This mirrors the morphometric trajectory analyses popularised by Frisoni et al.
(2010) and the conversion-prediction framework of Moradi et al. (2015), and
encodes the clinically established fact that *rate of atrophy*, not absolute
volume, is the single strongest non-invasive predictor of progression
[Diogo et al., 2022].

For *cross-sectional* subjects (OAS1_*) we have a single observation; longitudinal
deltas/slopes are encoded as zeros and a binary indicator ``is_longitudinal`` is
added so that the downstream learners can route attention accordingly. This
preserves the full sample size (∼800 subjects) while letting the gradient
boosters discover that slope features carry information only when
``is_longitudinal == 1``.

Group identifier
----------------
The CV ``groups`` vector returned alongside ``X`` is the ``subject_id``. All
splitters used downstream are *group-aware* (StratifiedGroupKFold), guaranteeing
that no subject contributes rows to both a training and a validation fold —
the leakage failure mode emphasised by Diogo et al. (2022).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import AegisConfig
from .data_loader import OasisFrames


@dataclass
class EngineeredDataset:
    """Bundle of design matrix, label vector, group vector and column metadata."""

    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    numeric_features: List[str]
    categorical_features: List[str]
    feature_names: List[str]


class FeatureEngineer:
    """Build the unified design matrix from harmonised OASIS frames."""

    # NB: CDR is *deliberately excluded* from the longitudinal dynamics — the
    # binary label is itself a function of CDR, so any derived CDR feature
    # (delta, slope, max) would leak the target into the design matrix.
    _MARKERS_FOR_DYNAMICS = ("nwbv", "mmse", "etiv", "asf")

    def __init__(self, cfg: AegisConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ public
    def build(self, frames: OasisFrames) -> EngineeredDataset:
        long_agg = self._aggregate_longitudinal(frames.longitudinal)
        cross_agg = self._prepare_cross_sectional(frames.cross_sectional)

        unified = pd.concat([cross_agg, long_agg], axis=0, ignore_index=True)
        unified = unified.dropna(subset=["cdr_label"]).reset_index(drop=True)

        y = self._build_target(unified["cdr_label"].to_numpy())
        groups = unified["subject_id"].to_numpy()

        feature_cols = [
            c for c in unified.columns if c not in {"subject_id", "cdr_label"}
        ]
        X = unified[feature_cols].copy()

        numeric = [c for c in feature_cols if c not in {"sex", "hand", "cohort"}]
        categorical = [c for c in feature_cols if c in {"sex", "hand", "cohort"}]

        return EngineeredDataset(
            X=X,
            y=y,
            groups=groups,
            numeric_features=numeric,
            categorical_features=categorical,
            feature_names=feature_cols,
        )

    # ------------------------------------------------------------ longitudinal
    def _aggregate_longitudinal(self, df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for subject_id, sub in df.groupby("subject_id"):
            sub = sub.sort_values("visit")
            baseline = sub.iloc[0]

            row: Dict[str, float] = {
                "subject_id": subject_id,
                "sex": baseline["sex"],
                "hand": baseline["hand"],
                "age": float(baseline["age"]),
                "educ": float(baseline["educ"]),
                "ses": float(baseline["ses"]) if pd.notna(baseline["ses"]) else np.nan,
                "mmse": float(baseline["mmse"]),
                "etiv": float(baseline["etiv"]),
                "nwbv": float(baseline["nwbv"]),
                "asf": float(baseline["asf"]),
                "cohort": "long",
                "is_longitudinal": 1,
                "n_visits": int(len(sub)),
                "followup_days": float(sub["mr_delay"].max()),
            }

            # Δ and OLS-slope features per marker
            for marker in self._MARKERS_FOR_DYNAMICS:
                series = sub[marker].astype(float).to_numpy()
                t = sub["mr_delay"].astype(float).to_numpy()
                row[f"{marker}_delta"] = self._delta(series)
                row[f"{marker}_slope"] = self._slope(t, series)
                row[f"{marker}_max"] = (
                    float(np.nanmax(series)) if np.any(~np.isnan(series)) else np.nan
                )

            # Label = max-CDR observed during follow-up (clinically: "ever-demented")
            row["cdr_label"] = float(np.nanmax(sub["cdr"].to_numpy()))
            records.append(row)
        return pd.DataFrame.from_records(records)

    # --------------------------------------------------------- cross-sectional
    def _prepare_cross_sectional(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "subject_id": df["subject_id"].to_numpy(),
                "sex": df["sex"].to_numpy(),
                "hand": df["hand"].to_numpy(),
                "age": df["age"].astype(float).to_numpy(),
                "educ": df["educ"].astype(float).to_numpy(),
                "ses": df["ses"].astype(float).to_numpy(),
                "mmse": df["mmse"].astype(float).to_numpy(),
                "etiv": df["etiv"].astype(float).to_numpy(),
                "nwbv": df["nwbv"].astype(float).to_numpy(),
                "asf": df["asf"].astype(float).to_numpy(),
                "cohort": "cross",
                "is_longitudinal": 0,
                "n_visits": 1,
                "followup_days": 0.0,
            }
        )

        # Δ / slope features are undefined for single-visit subjects; we encode
        # them as 0.0 and rely on the ``is_longitudinal`` indicator to neutralise
        # them at the model level. Imputation is not applied to these columns
        # because zero is the *semantically correct* value (no observed change).
        for marker in self._MARKERS_FOR_DYNAMICS:
            out[f"{marker}_delta"] = 0.0
            out[f"{marker}_slope"] = 0.0
            out[f"{marker}_max"] = df[marker].astype(float).to_numpy()

        out["cdr_label"] = df["cdr"].astype(float).to_numpy()
        return out

    # ------------------------------------------------------------------ target
    def _build_target(self, cdr: np.ndarray) -> np.ndarray:
        if self.cfg.target_mode == "binary":
            return (cdr >= self.cfg.binary_threshold).astype(int)
        if self.cfg.target_mode == "multiclass":
            mapping = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
            return np.array([mapping.get(float(v), 0) for v in cdr], dtype=int)
        raise ValueError(f"Unknown target_mode: {self.cfg.target_mode}")

    # -------------------------------------------------------- numeric helpers
    @staticmethod
    def _delta(series: np.ndarray) -> float:
        valid = series[~np.isnan(series)]
        if valid.size < 2:
            return 0.0
        return float(valid[-1] - valid[0])

    @staticmethod
    def _slope(t: np.ndarray, y: np.ndarray) -> float:
        mask = ~(np.isnan(t) | np.isnan(y))
        if mask.sum() < 2:
            return 0.0
        t_valid, y_valid = t[mask], y[mask]
        if np.allclose(t_valid, t_valid[0]):
            return 0.0
        # Closed-form OLS slope: cov(t,y) / var(t)
        return float(np.cov(t_valid, y_valid, ddof=0)[0, 1] / np.var(t_valid))
