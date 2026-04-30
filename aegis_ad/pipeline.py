"""End-to-end Aegis-AD orchestrator.

Outer ``StratifiedGroupKFold`` cross-validation drives the entire workflow:

    For each outer fold (subjects partitioned into train / held-out):
        1. Fit the leak-free preprocessor on training subjects only.
        2. Apply SMOTE on the *transformed* training set (validation untouched).
        3. Tune each tree booster with Optuna over an inner GroupKFold.
        4. Build and fit the AegisEnsemble on the full training partition.
        5. Score the held-out subjects and accumulate per-fold metrics.
    Refit on all data and persist:
        - The fitted preprocessor + ensemble (joblib).
        - SHAP global plots and local subject explanations.
        - Metric tables and ROC / PR curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedGroupKFold

from .config import AegisConfig
from .data_loader import DataLoader
from .evaluation import EvaluationReport, Evaluator
from .explainability import ShapExplainer
from .feature_engineer import EngineeredDataset, FeatureEngineer
from .models.ensemble import AegisEnsemble
from .preprocessor import build_preprocessor, resolve_feature_names
from .tuning import OptunaTuner


@dataclass
class PipelineArtifacts:
    """Filesystem paths of the artifacts emitted by a successful run."""

    metrics_csv: Path
    metrics_json: Path
    roc_pr_png: Path
    model_joblib: Path
    preprocessor_joblib: Path
    shap_dir: Path


class AegisPipeline:
    """Top-level orchestrator. The only public entry point is ``run``."""

    def __init__(self, cfg: AegisConfig) -> None:
        self.cfg = cfg
        self.loader = DataLoader(cfg)
        self.engineer = FeatureEngineer(cfg)
        self.tuner = OptunaTuner(cfg)
        self.evaluator = Evaluator()

    # =================================================================== run
    def run(self) -> PipelineArtifacts:
        out = self.cfg.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # ------------------------ 1. data + features ----------------------
        frames = self.loader.load()
        ds: EngineeredDataset = self.engineer.build(frames)
        print(
            f"[aegis] design matrix: {ds.X.shape},  positives: {ds.y.sum()} / {len(ds.y)}"
        )

        # ------------------------ 2. outer CV -----------------------------
        outer = StratifiedGroupKFold(
            n_splits=self.cfg.n_splits,
            shuffle=True,
            random_state=self.cfg.random_state,
        )
        report = EvaluationReport()

        for fold_idx, (tr, te) in enumerate(
            outer.split(ds.X, ds.y, ds.groups), start=1
        ):
            print(f"[aegis] ── outer fold {fold_idx}/{self.cfg.n_splits} ──")
            X_tr_df, X_te_df = ds.X.iloc[tr], ds.X.iloc[te]
            y_tr, y_te = ds.y[tr], ds.y[te]
            g_tr = ds.groups[tr]

            # -- 2.a fit preprocessor on training fold only -----------------
            prep, _ = build_preprocessor(ds.numeric_features, ds.categorical_features)
            X_tr = prep.fit_transform(X_tr_df)
            X_te = prep.transform(X_te_df)

            # -- 2.b SMOTE on transformed training fold ---------------------
            X_tr_bal, y_tr_bal, g_tr_bal = self._maybe_smote(X_tr, y_tr, g_tr)

            spw = self._scale_pos_weight(y_tr_bal)

            # -- 2.c Optuna tuning (inner group-aware CV) -------------------
            tuned = self.tuner.tune(X_tr_bal, y_tr_bal, g_tr_bal, scale_pos_weight=spw)
            print(f"[aegis] tuned params: {tuned}")

            # -- 2.d Build & fit Aegis ensemble -----------------------------
            ensemble = AegisEnsemble(random_state=self.cfg.random_state).build(
                n_splits=self.cfg.inner_n_splits,
                scale_pos_weight=spw,
                params=tuned,
            )
            ensemble.fit(X_tr_bal, y_tr_bal, groups=g_tr_bal)

            # -- 2.e Score held-out subjects --------------------------------
            proba_te = ensemble.predict_proba(X_te)[:, 1]
            fold_result = self.evaluator.evaluate(fold_idx, y_te, proba_te)
            report.append(fold_result)
            print(f"[aegis] fold {fold_idx} metrics: {fold_result.metrics}")

        # ------------------------ 3. persist CV report --------------------
        report.save(out)
        self.evaluator.plot_curves(report, out)

        # ------------------------ 4. final refit on all data --------------
        prep_final, _ = build_preprocessor(ds.numeric_features, ds.categorical_features)
        X_all = prep_final.fit_transform(ds.X)
        feature_names = resolve_feature_names(prep_final)

        X_all_bal, y_all_bal, g_all_bal = self._maybe_smote(X_all, ds.y, ds.groups)
        spw_all = self._scale_pos_weight(y_all_bal)
        tuned_all = self.tuner.tune(
            X_all_bal, y_all_bal, g_all_bal, scale_pos_weight=spw_all
        )
        final_ensemble = AegisEnsemble(random_state=self.cfg.random_state).build(
            n_splits=self.cfg.inner_n_splits,
            scale_pos_weight=spw_all,
            params=tuned_all,
        )
        final_ensemble.fit(X_all_bal, y_all_bal, groups=g_all_bal)

        # ------------------------ 5. SHAP explanations --------------------
        shap_dir = out / "shap"
        explainer = ShapExplainer(self.cfg, feature_names=feature_names)
        focal = X_all[: self.cfg.shap_local_examples]
        explainer.explain(final_ensemble, X_all, focal, shap_dir)

        # ------------------------ 6. persist model ------------------------
        model_path = out / "aegis_ensemble.joblib"
        prep_path = out / "preprocessor.joblib"
        joblib.dump(final_ensemble, model_path)
        joblib.dump(prep_final, prep_path)

        print("[aegis] ── final summary ──")
        print(report.summary().to_string())

        return PipelineArtifacts(
            metrics_csv=out / "metrics_per_fold.csv",
            metrics_json=out / "metrics_summary.json",
            roc_pr_png=out / "roc_pr_curves.png",
            model_joblib=model_path,
            preprocessor_joblib=prep_path,
            shap_dir=shap_dir,
        )

    # ============================================================= helpers
    def _maybe_smote(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply SMOTE if enabled and the minority class is sufficiently small.

        Synthetic samples are appended at the end of the array; the corresponding
        ``groups`` entries are filled with synthetic identifiers (negative
        integers) so they cannot collide with real subject IDs but still satisfy
        the group-aware CV interface in downstream tuning.
        """
        if not self.cfg.use_smote:
            return X, y, groups
        n_pos = int(y.sum())
        n_neg = int((1 - y).sum())
        if min(n_pos, n_neg) <= self.cfg.smote_k_neighbors:
            return X, y, groups
        sm = SMOTE(
            random_state=self.cfg.random_state,
            k_neighbors=self.cfg.smote_k_neighbors,
        )
        X_res, y_res = sm.fit_resample(X, y)
        n_synth = len(y_res) - len(y)
        synth_groups = np.array([f"_synth_{i}" for i in range(n_synth)])
        groups_res = np.concatenate([groups.astype(str), synth_groups])
        return X_res, y_res, groups_res

    @staticmethod
    def _scale_pos_weight(y: np.ndarray) -> float:
        n_pos = max(int(y.sum()), 1)
        n_neg = max(int((1 - y).sum()), 1)
        return float(n_neg / n_pos)
