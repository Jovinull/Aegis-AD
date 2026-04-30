"""Data ingestion for the OASIS cross-sectional and longitudinal cohorts.

The two CSV exports use slightly inconsistent column names ("Educ" vs. "EDUC",
"ID" vs. "Subject ID" / "MRI ID", etc.). The loader harmonises both schemas to a
single canonical naming convention so downstream feature engineering can treat
both cohorts uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .config import AegisConfig


@dataclass
class OasisFrames:
    """Container for the two harmonised cohort DataFrames."""

    cross_sectional: pd.DataFrame
    longitudinal: pd.DataFrame


class DataLoader:
    """Load and harmonise the OASIS-1 and OASIS-2 CSV files.

    The harmonised schema is::

        ['subject_id', 'mri_id', 'visit', 'mr_delay', 'group',
         'sex', 'hand', 'age', 'educ', 'ses',
         'mmse', 'cdr', 'etiv', 'nwbv', 'asf', 'cohort']

    where ``cohort`` ∈ {"cross", "long"}. Cross-sectional rows receive a synthetic
    ``visit = 1`` and ``mr_delay = 0`` so that downstream temporal aggregation
    can operate on a uniform schema.
    """

    _RENAME_CROSS = {
        "ID": "mri_id",
        "M/F": "sex",
        "Hand": "hand",
        "Age": "age",
        "Educ": "educ",
        "SES": "ses",
        "MMSE": "mmse",
        "CDR": "cdr",
        "eTIV": "etiv",
        "nWBV": "nwbv",
        "ASF": "asf",
        "Delay": "mr_delay",
    }

    _RENAME_LONG = {
        "Subject ID": "subject_id",
        "MRI ID": "mri_id",
        "Group": "group",
        "Visit": "visit",
        "MR Delay": "mr_delay",
        "M/F": "sex",
        "Hand": "hand",
        "Age": "age",
        "EDUC": "educ",
        "SES": "ses",
        "MMSE": "mmse",
        "CDR": "cdr",
        "eTIV": "etiv",
        "nWBV": "nwbv",
        "ASF": "asf",
    }

    def __init__(self, cfg: AegisConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ public
    def load(self) -> OasisFrames:
        cross = self._load_cross_sectional(self.cfg.cross_sectional_path)
        long = self._load_longitudinal(self.cfg.longitudinal_path)
        return OasisFrames(cross_sectional=cross, longitudinal=long)

    # ------------------------------------------------------------------ private
    def _load_cross_sectional(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, na_values=["N/A", "NA", "", " "])
        df = df.rename(columns=self._RENAME_CROSS)
        # OASIS-1 IDs are e.g. "OAS1_0001_MR1"; subject_id is everything but the
        # trailing "_MRn" suffix. Each cross-sectional subject contributes a
        # single observation, so subject_id == mri_id-prefix uniquely identifies
        # the patient and is therefore a valid CV group.
        df["subject_id"] = df["mri_id"].str.replace(r"_MR\d+$", "", regex=True)
        df["visit"] = 1
        df["mr_delay"] = 0
        df["group"] = np.nan  # OASIS-1 has no longitudinal Group field
        df["cohort"] = "cross"
        df = self._coerce_numeric(df)
        df = df.dropna(subset=["cdr"]).reset_index(drop=True)
        return df

    def _load_longitudinal(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, na_values=["N/A", "NA", "", " "])
        df = df.rename(columns=self._RENAME_LONG)
        df["cohort"] = "long"
        df = self._coerce_numeric(df)
        df = df.dropna(subset=["cdr"]).reset_index(drop=True)
        df = df.sort_values(["subject_id", "visit"]).reset_index(drop=True)
        return df

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = ["age", "educ", "ses", "mmse", "cdr",
                        "etiv", "nwbv", "asf", "mr_delay", "visit"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
