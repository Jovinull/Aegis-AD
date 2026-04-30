"""Comprehensive evaluation suite for clinical classifiers.

Reports the metrics expected by neuroimaging journals (Diogo et al., 2022;
AlSaeed & Omar, 2022): ROC-AUC, PR-AUC, F1, balanced accuracy, sensitivity
(recall on the positive class), specificity (recall on the negative class), and
Matthews correlation coefficient (a robust single-number summary for imbalanced
binary tasks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class FoldResult:
    fold: int
    metrics: Dict[str, float]
    confusion: np.ndarray
    roc: Dict[str, np.ndarray]
    pr: Dict[str, np.ndarray]


@dataclass
class EvaluationReport:
    fold_results: List[FoldResult] = field(default_factory=list)

    def append(self, result: FoldResult) -> None:
        self.fold_results.append(result)

    def summary(self) -> pd.DataFrame:
        if not self.fold_results:
            return pd.DataFrame()
        df = pd.DataFrame([r.metrics for r in self.fold_results])
        df.index = [f"fold_{r.fold}" for r in self.fold_results]
        df.loc["mean"] = df.mean(numeric_only=True)
        df.loc["std"] = df.std(numeric_only=True, ddof=1)
        return df.round(4)

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.summary().to_csv(output_dir / "metrics_per_fold.csv")
        with (output_dir / "metrics_summary.json").open("w") as fh:
            mean_metrics = self.summary().loc["mean"].to_dict()
            json.dump(mean_metrics, fh, indent=2)


class Evaluator:
    """Compute the standard clinical metric suite for a binary classifier."""

    def __init__(self, decision_threshold: float = 0.5) -> None:
        self.decision_threshold = decision_threshold

    # ------------------------------------------------------------------ public
    def evaluate(
        self, fold: int, y_true: np.ndarray, y_proba: np.ndarray
    ) -> FoldResult:
        y_true = np.asarray(y_true).astype(int).ravel()
        y_proba = np.asarray(y_proba).astype(float).ravel()
        y_pred = (y_proba >= self.decision_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)

        metrics = {
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "pr_auc": float(average_precision_score(y_true, y_proba)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)

        return FoldResult(
            fold=fold,
            metrics=metrics,
            confusion=np.array([[tn, fp], [fn, tp]]),
            roc={"fpr": fpr, "tpr": tpr},
            pr={"precision": prec_curve, "recall": rec_curve},
        )

    # ----------------------------------------------------------------- plots
    def plot_curves(self, report: EvaluationReport, output_dir: Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            return
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for r in report.fold_results:
            axes[0].plot(
                r.roc["fpr"],
                r.roc["tpr"],
                label=f"fold {r.fold} (AUC={r.metrics['roc_auc']:.3f})",
            )
            axes[1].plot(
                r.pr["recall"],
                r.pr["precision"],
                label=f"fold {r.fold} (AP={r.metrics['pr_auc']:.3f})",
            )
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="ROC — outer CV",
        )
        axes[1].set(
            xlabel="Recall", ylabel="Precision", title="Precision–Recall — outer CV"
        )
        for ax in axes:
            ax.legend(loc="best", fontsize=8)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "roc_pr_curves.png", dpi=150)
        plt.close(fig)
