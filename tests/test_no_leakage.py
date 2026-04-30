"""Leakage-prevention test suite for Aegis-AD.

These tests are the *scientific contract* of the project: each one encodes a
guarantee that, if violated, would invalidate every metric subsequently
reported in the paper. They run against a fully synthetic, deterministic
miniature of the OASIS-1 / OASIS-2 cohorts so they execute in seconds and
require no patient data.

Mathematical contracts asserted here
------------------------------------
1. **Subject-level isolation** — for every outer fold ``k``::

        SubjectsTrain(k)  ∩  SubjectsValidation(k)  ≡  ∅

2. **SMOTE containment** — synthetic minority-class samples carry a reserved
   identifier prefix that cannot collide with any real ``Subject ID`` and
   never appear in the held-out validation partition.
3. **Causal feature engineering** — the longitudinal slope operator is a pure
   function of the visit window supplied; it cannot be influenced by future
   visits, by other subjects, or by global state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from aegis_ad.config import AegisConfig
from aegis_ad.data_loader import DataLoader, OasisFrames
from aegis_ad.feature_engineer import FeatureEngineer

_SEED = 1234


# ============================================================================
# Synthetic OASIS-2 / OASIS-1 fixtures
# ============================================================================
def _build_synthetic_longitudinal() -> pd.DataFrame:
    """Construct a deterministic OASIS-2-shaped longitudinal cohort.

    Population:
        * 50 subjects
        * 20 with a single visit, 20 with two visits, 10 with three visits
        * ≈ 40 % positive class (CDR ≥ 0.5)

    Each visit carries plausible nWBV/eTIV/MMSE values and a monotonically
    progressing ``MR Delay`` so that slope features can be exercised. The
    column names mirror the real OASIS-2 export so the production
    ``DataLoader`` exercises its full harmonisation path on the synthetic
    CSV.
    """
    rng = np.random.default_rng(_SEED)
    visit_counts: List[int] = ([1] * 20) + ([2] * 20) + ([3] * 10)
    rng.shuffle(visit_counts)

    rows: List[Dict] = []
    for i, n_visits in enumerate(visit_counts):
        sid = f"OAS2_{i:04d}"
        is_demented = bool(rng.random() < 0.4)
        sex = str(rng.choice(["M", "F"]))
        age0 = int(rng.integers(60, 90))
        educ = int(rng.integers(8, 20))
        ses = int(rng.integers(1, 5))

        nwbv0 = float(rng.normal(0.70 if is_demented else 0.74, 0.018))
        etiv0 = float(rng.normal(1500, 100))
        mmse0 = int(rng.integers(18, 27)) if is_demented else int(rng.integers(26, 30))
        cdr0 = 0.5 if is_demented else 0.0

        for v in range(n_visits):
            mr_delay = int(v * rng.integers(180, 540))
            nwbv = nwbv0 - 0.005 * v + float(rng.normal(0, 0.003))
            mmse = max(0, mmse0 - (1 if is_demented else 0) * v)
            asf = 1.0 / (etiv0 / 1500.0)
            rows.append(
                {
                    "Subject ID": sid,
                    "MRI ID": f"{sid}_MR{v + 1}",
                    "Group": "Demented" if is_demented else "Nondemented",
                    "Visit": v + 1,
                    "MR Delay": mr_delay,
                    "M/F": sex,
                    "Hand": "R",
                    "Age": age0 + v,
                    "EDUC": educ,
                    "SES": ses,
                    "MMSE": mmse,
                    "CDR": cdr0,
                    "eTIV": etiv0,
                    "nWBV": nwbv,
                    "ASF": asf,
                }
            )
    return pd.DataFrame(rows)


def _build_synthetic_cross_sectional() -> pd.DataFrame:
    """30 subjects in OASIS-1 schema; ≈ 35 % positive class."""
    rng = np.random.default_rng(_SEED + 1)
    rows: List[Dict] = []
    for i in range(30):
        is_demented = bool(rng.random() < 0.35)
        rows.append(
            {
                "ID": f"OAS1_{i:04d}_MR1",
                "M/F": str(rng.choice(["M", "F"])),
                "Hand": "R",
                "Age": int(rng.integers(55, 95)),
                "Educ": int(rng.integers(8, 20)),
                "SES": int(rng.integers(1, 5)),
                "MMSE": (
                    int(rng.integers(18, 27))
                    if is_demented
                    else int(rng.integers(27, 31))
                ),
                "CDR": 0.5 if is_demented else 0.0,
                "eTIV": float(rng.normal(1500, 100)),
                "nWBV": float(rng.normal(0.71 if is_demented else 0.75, 0.02)),
                "ASF": float(rng.normal(1.2, 0.1)),
                "Delay": "N/A",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory) -> Path:
    """Persist synthetic CSVs in a per-session temp directory."""
    path = tmp_path_factory.mktemp("oasis_synth")
    _build_synthetic_longitudinal().to_csv(path / "oasis_longitudinal.csv", index=False)
    _build_synthetic_cross_sectional().to_csv(
        path / "oasis_cross-sectional.csv", index=False
    )
    return path


@pytest.fixture(scope="session")
def synthetic_config(synthetic_data_dir: Path) -> AegisConfig:
    return AegisConfig(
        data_dir=synthetic_data_dir,
        n_splits=5,
        inner_n_splits=3,
        random_state=_SEED,
        use_smote=True,
        smote_k_neighbors=3,
    )


@pytest.fixture(scope="session")
def synthetic_frames(synthetic_config: AegisConfig) -> OasisFrames:
    return DataLoader(synthetic_config).load()


@pytest.fixture(scope="session")
def synthetic_dataset(synthetic_config: AegisConfig, synthetic_frames: OasisFrames):
    return FeatureEngineer(synthetic_config).build(synthetic_frames)


# ============================================================================
# Test 1 — strict subject-level isolation in cross-validation
# ============================================================================
class TestStrictGroupIsolation:
    """Train and validation partitions of every fold must be disjoint at the
    subject level. This is the cornerstone of any honest longitudinal AD
    benchmark and the failure mode flagged by Diogo et al. (2022)."""

    def test_strict_group_isolation_in_cv(
        self, synthetic_dataset, synthetic_config: AegisConfig
    ):
        cv = StratifiedGroupKFold(
            n_splits=synthetic_config.n_splits,
            shuffle=True,
            random_state=synthetic_config.random_state,
        )
        for fold, (tr, va) in enumerate(
            cv.split(
                synthetic_dataset.X,
                synthetic_dataset.y,
                synthetic_dataset.groups,
            ),
            start=1,
        ):
            train_subjects = set(synthetic_dataset.groups[tr].tolist())
            val_subjects = set(synthetic_dataset.groups[va].tolist())
            intersection = train_subjects & val_subjects
            assert intersection == set(), (
                f"Fold {fold}: {len(intersection)} subjects appear in both "
                f"train and validation: {sorted(intersection)[:5]}"
            )

    def test_every_subject_validated_exactly_once(
        self, synthetic_dataset, synthetic_config: AegisConfig
    ):
        """Each subject must contribute to exactly one validation fold —
        violations indicate either CV mis-configuration or duplicate group
        identifiers in the design matrix."""
        cv = StratifiedGroupKFold(
            n_splits=synthetic_config.n_splits,
            shuffle=True,
            random_state=synthetic_config.random_state,
        )
        validation_folds: Dict[str, List[int]] = {}
        for fold, (_, va) in enumerate(
            cv.split(
                synthetic_dataset.X,
                synthetic_dataset.y,
                synthetic_dataset.groups,
            ),
            start=1,
        ):
            for sid in synthetic_dataset.groups[va]:
                validation_folds.setdefault(sid, []).append(fold)

        for sid, folds in validation_folds.items():
            assert (
                len(folds) == 1
            ), f"Subject {sid} validated in folds {folds}; expected exactly 1."
        all_subjects = set(synthetic_dataset.groups.tolist())
        assert set(validation_folds) == all_subjects, (
            f"{len(all_subjects - set(validation_folds))} subject(s) never "
            "appeared in a validation fold."
        )


# ============================================================================
# Test 2 — SMOTE leakage prevention
# ============================================================================
class TestSmoteLeakagePrevention:
    """Synthetic SMOTE samples must carry a reserved sentinel identifier so
    they cannot be confused with real subjects, and cannot collide with any
    held-out validation Subject ID."""

    def test_smote_leakage_prevention(
        self, synthetic_dataset, synthetic_config: AegisConfig
    ):
        pytest.importorskip("imblearn")
        from aegis_ad.pipeline import AegisPipeline
        from aegis_ad.preprocessor import build_preprocessor

        cv = StratifiedGroupKFold(
            n_splits=synthetic_config.n_splits,
            shuffle=True,
            random_state=synthetic_config.random_state,
        )
        tr, va = next(
            iter(
                cv.split(
                    synthetic_dataset.X,
                    synthetic_dataset.y,
                    synthetic_dataset.groups,
                )
            )
        )

        prep, _ = build_preprocessor(
            synthetic_dataset.numeric_features,
            synthetic_dataset.categorical_features,
        )
        X_tr = prep.fit_transform(synthetic_dataset.X.iloc[tr])
        y_tr = synthetic_dataset.y[tr]
        g_tr = synthetic_dataset.groups[tr]

        pipeline = AegisPipeline(synthetic_config)
        X_bal, y_bal, g_bal = pipeline._maybe_smote(X_tr, y_tr, g_tr)

        n_synth = len(g_bal) - len(g_tr)
        if n_synth == 0:
            pytest.skip("Class distribution did not trigger SMOTE.")

        synth_ids = list(g_bal[len(g_tr) :])
        real_ids = set(synthetic_dataset.groups.astype(str).tolist())
        val_ids = set(synthetic_dataset.groups[va].astype(str).tolist())

        # 1. Sentinel prefix
        assert all(
            str(s).startswith("_synth_") for s in synth_ids
        ), "Synthetic SMOTE identifiers must start with the '_synth_' prefix."
        # 2. Disjoint from every real subject in the cohort
        assert real_ids.isdisjoint(
            set(map(str, synth_ids))
        ), "A synthetic identifier collides with a real Subject ID."
        # 3. Disjoint from the held-out validation subjects (the strongest
        #    statement of containment for outer-CV reporting integrity)
        assert val_ids.isdisjoint(set(map(str, synth_ids))), (
            "Synthetic SMOTE samples bled into the validation fold's " "subject space."
        )
        # 4. Class balance must be improved (or at minimum preserved)
        assert int(y_bal.sum()) >= int(
            y_tr.sum()
        ), "SMOTE must not reduce the minority-class count."
        # 5. Every synthetic id is unique (one synthetic group per row)
        assert len(synth_ids) == len(
            set(synth_ids)
        ), "Synthetic identifiers must be globally unique."


# ============================================================================
# Test 3 — temporal feature engineering must not look at the future
# ============================================================================
class TestTemporalNoLookAhead:
    """Slopes computed for a window [v_1, …, v_k] must depend only on those
    visits — never on visits v_{k+1..n}, never on other subjects, never on
    global state."""

    def test_slope_invariant_under_future_modifications(self):
        """Identical past+present (visits 1..3) must yield identical slope,
        even when visits 4..5 differ arbitrarily."""
        t = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
        y_a = np.array([0.75, 0.74, 0.72, 0.50, 0.30])  # divergent suffix
        y_b = np.array([0.75, 0.74, 0.72, 0.69, 0.65])  # gentler suffix

        s_a_prefix = FeatureEngineer._slope(t[:3], y_a[:3])
        s_b_prefix = FeatureEngineer._slope(t[:3], y_b[:3])
        assert s_a_prefix == pytest.approx(s_b_prefix, abs=1e-12), (
            "Prefix-window slope was influenced by future visits — temporal " "leakage."
        )
        # Sanity: if we *do* include the divergent suffix, slopes must differ
        s_a_full = FeatureEngineer._slope(t, y_a)
        s_b_full = FeatureEngineer._slope(t, y_b)
        assert s_a_full != pytest.approx(s_b_full)

    def test_slope_recovers_known_linear_trend(self):
        """The OLS estimator must return the analytic slope on noiseless
        linear data — a numerical regression guard."""
        t = np.linspace(0.0, 1000.0, 6)
        y = 0.80 - 1e-4 * t
        s = FeatureEngineer._slope(t, y)
        assert s == pytest.approx(-1e-4, rel=1e-9, abs=1e-12)

    def test_slope_handles_degenerate_inputs(self):
        """Single observation, all-NaN, and zero-variance time vectors must
        return 0.0 rather than NaN/Inf, otherwise downstream imputation
        would receive non-finite values."""
        assert FeatureEngineer._slope(np.array([100.0]), np.array([0.74])) == 0.0
        assert (
            FeatureEngineer._slope(np.array([0.0, 100.0]), np.array([np.nan, np.nan]))
            == 0.0
        )
        assert (
            FeatureEngineer._slope(np.array([100.0, 100.0]), np.array([0.74, 0.71]))
            == 0.0
        )

    def test_subject_features_invariant_to_other_subjects(
        self, synthetic_config: AegisConfig, synthetic_frames: OasisFrames
    ):
        """Features for subject S must depend only on S's own visits.
        Re-running the engineer on the *same* subject in isolation must
        yield byte-identical features to the full-cohort run."""
        full_ds = FeatureEngineer(synthetic_config).build(synthetic_frames)

        # Pick a long-cohort subject with > 1 visit so slope features are
        # actually exercised.
        long_df = synthetic_frames.longitudinal
        candidate = (
            long_df.groupby("subject_id").size().pipe(lambda s: s[s >= 2]).index[0]
        )
        full_idx = int(np.where(full_ds.groups == candidate)[0][0])
        full_row = full_ds.X.iloc[full_idx]

        isolated_long = long_df[long_df["subject_id"] == candidate].copy()
        isolated_cross = synthetic_frames.cross_sectional.iloc[0:0].copy()
        isolated_frames = OasisFrames(
            cross_sectional=isolated_cross, longitudinal=isolated_long
        )
        isolated_ds = FeatureEngineer(synthetic_config).build(isolated_frames)
        assert len(isolated_ds.groups) == 1
        isolated_row = isolated_ds.X.iloc[0]

        for col in full_row.index:
            lhs, rhs = full_row[col], isolated_row[col]
            if isinstance(lhs, float) and isinstance(rhs, float):
                assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12), (
                    f"Column '{col}' differs across cohort vs. isolated build "
                    f"({lhs} vs. {rhs}) — cross-subject contamination."
                )
            else:
                assert lhs == rhs, (
                    f"Column '{col}' differs across cohort vs. isolated build "
                    f"({lhs!r} vs. {rhs!r}) — cross-subject contamination."
                )


# ============================================================================
# Test 4 — preprocessor refits per fold (no shared statistics)
# ============================================================================
class TestPreprocessorPerFoldRefit:
    """The KNNImputer / RobustScaler must be refit per training fold; if the
    same statistics appear across folds we are silently reusing global
    information — a subtle but real form of leakage."""

    def test_robust_scaler_centres_change_between_folds(
        self, synthetic_dataset, synthetic_config: AegisConfig
    ):
        from aegis_ad.preprocessor import build_preprocessor

        cv = StratifiedGroupKFold(
            n_splits=synthetic_config.n_splits,
            shuffle=True,
            random_state=synthetic_config.random_state,
        )
        centres = []
        for tr, _ in cv.split(
            synthetic_dataset.X, synthetic_dataset.y, synthetic_dataset.groups
        ):
            prep, _ = build_preprocessor(
                synthetic_dataset.numeric_features,
                synthetic_dataset.categorical_features,
            )
            prep.fit(synthetic_dataset.X.iloc[tr])
            scaler = prep.named_transformers_["num"].named_steps["scaler"]
            centres.append(np.asarray(scaler.center_))

        any_diff = any(
            not np.allclose(centres[i], centres[j])
            for i in range(len(centres))
            for j in range(i + 1, len(centres))
        )
        assert any_diff, (
            "RobustScaler.center_ is identical across all folds — this "
            "implies a shared (i.e. leaked) global statistic."
        )


# ============================================================================
# Test 5 — design matrix excludes target-derived features
# ============================================================================
def test_no_cdr_derived_features_in_design_matrix(synthetic_dataset):
    """`cdr*` features would constitute outright target leakage because the
    binary label is itself a thresholding of CDR."""
    leaky = [c for c in synthetic_dataset.feature_names if "cdr" in c.lower()]
    assert not leaky, (
        f"CDR-derived columns must be excluded from the design matrix; "
        f"found: {leaky}"
    )
