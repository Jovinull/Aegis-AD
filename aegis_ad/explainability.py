"""SHAP-based clinical explainability for the Aegis ensemble.

The explainer computes Shapley contributions in two complementary regimes:

1. **Per-branch tree explanations** (``shap.TreeExplainer``) for the gradient
   boosters. These are exact for tree ensembles (Lundberg et al., 2020) and
   yield the global feature-importance plots required by the audit narrative
   ("does the model actually rely on hippocampal/atrophy proxies and cognitive
   decline rate, as the literature predicts?").

2. **Model-agnostic ensemble explanations** (``shap.KernelExplainer``) for the
   *full* stacked model. Kernel SHAP is expensive, so we restrict it to a
   capped background set (``cfg.shap_max_samples``) and a small number of
   focal subjects (``cfg.shap_local_examples``) for which we generate local
   force-style summaries — directly addressing the explainability requirement
   articulated in the Aegis-AD specification.

References
----------
Lundberg, S.M. et al. (2020) *From local explanations to global understanding
  with explainable AI for trees*. Nature Machine Intelligence 2, 56–67.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from .config import AegisConfig
from .models.ensemble import GroupAwareStackingClassifier


class ShapExplainer:
    """Generate global and local SHAP explanations for the Aegis ensemble."""

    def __init__(self, cfg: AegisConfig, feature_names: Sequence[str]) -> None:
        self.cfg = cfg
        self.feature_names = list(feature_names)

    # ------------------------------------------------------------------ public
    def explain(
        self,
        ensemble: GroupAwareStackingClassifier,
        X_background: np.ndarray,
        X_focal: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Persist global summary plots and per-subject local explanations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            return

        bg = self._cap(X_background, self.cfg.shap_max_samples)

        # -- 1. Global per-branch tree SHAP ------------------------------
        global_records: List[dict] = []
        for name, est in ensemble.fitted_base_:
            if name == "tabnet":
                continue
            try:
                explainer = shap.TreeExplainer(
                    est, bg, feature_perturbation="interventional"
                )
                shap_values = explainer.shap_values(bg)
                if isinstance(shap_values, list):  # multi-class returns a list
                    shap_values = shap_values[1]
                fig = plt.figure(figsize=(9, 6))
                shap.summary_plot(
                    shap_values,
                    bg,
                    feature_names=self.feature_names,
                    show=False,
                    max_display=20,
                )
                plt.title(f"Global SHAP — branch '{name}'")
                plt.tight_layout()
                plt.savefig(output_dir / f"shap_global_{name}.png", dpi=150)
                plt.close()

                mean_abs = np.abs(shap_values).mean(axis=0)
                for fname, score in zip(self.feature_names, mean_abs):
                    global_records.append(
                        {
                            "branch": name,
                            "feature": fname,
                            "mean_abs_shap": float(score),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[shap] tree explanation failed for branch '{name}': {exc}")

        if global_records:
            (
                pd.DataFrame(global_records)
                .sort_values(["branch", "mean_abs_shap"], ascending=[True, False])
                .to_csv(output_dir / "shap_global_importance.csv", index=False)
            )

        # -- 2. Ensemble-level KernelExplainer for local rationales -------
        try:
            import shap

            focal = self._cap(X_focal, self.cfg.shap_local_examples)

            def _proba_positive(x: np.ndarray) -> np.ndarray:
                return ensemble.predict_proba(x)[:, 1]

            kernel = shap.KernelExplainer(_proba_positive, self._cap(bg, 100))
            shap_values = kernel.shap_values(focal, nsamples=200, silent=True)

            local_df = pd.DataFrame(shap_values, columns=self.feature_names)
            local_df.insert(0, "subject_index", np.arange(len(focal)))
            local_df.to_csv(output_dir / "shap_local_examples.csv", index=False)

            import matplotlib.pyplot as plt

            for i in range(len(focal)):
                contrib = shap_values[i]
                order = np.argsort(np.abs(contrib))[::-1][:12]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(
                    [self.feature_names[j] for j in order][::-1],
                    [contrib[j] for j in order][::-1],
                )
                ax.set_title(
                    f"Local SHAP — subject {i}  "
                    f"(P(AD) = {ensemble.predict_proba(focal[i:i+1])[0,1]:.2f})"
                )
                ax.axvline(0, color="k", lw=0.5)
                plt.tight_layout()
                plt.savefig(output_dir / f"shap_local_subject_{i}.png", dpi=150)
                plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            print(f"[shap] kernel explanation failed: {exc}")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _cap(X: np.ndarray, n: int) -> np.ndarray:
        if len(X) <= n:
            return X
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=n, replace=False)
        return X[idx]
