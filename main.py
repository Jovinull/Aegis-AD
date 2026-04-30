"""Aegis-AD entry point.

Usage
-----
    python main.py                   # default configuration
    python main.py --no-smote        # disable SMOTE
    python main.py --multiclass      # CDR ∈ {0, 0.5, 1, 2}
    python main.py --trials 60       # change Optuna budget

Outputs are written to ``artifacts/``:

    metrics_per_fold.csv      Per-fold + mean ± std of every metric
    metrics_summary.json      Mean metrics for paper inclusion
    roc_pr_curves.png         Outer-CV ROC and Precision–Recall curves
    aegis_ensemble.joblib     Refit-on-all stacking ensemble
    preprocessor.joblib       Fitted ColumnTransformer
    shap/                     Global + local SHAP explanations
"""
from __future__ import annotations

import argparse
from pathlib import Path

from aegis_ad.config import AegisConfig
from aegis_ad.pipeline import AegisPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aegis-AD pipeline runner.")
    p.add_argument("--data-dir", type=Path, default=Path("dataset"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--trials", type=int, default=40,
                   help="Total Optuna trials, split across base learners.")
    p.add_argument("--timeout", type=int, default=600,
                   help="Total Optuna wall-clock budget per fold (seconds).")
    p.add_argument("--no-smote", action="store_true")
    p.add_argument("--multiclass", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AegisConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        n_optuna_trials=args.trials,
        optuna_timeout=args.timeout,
        use_smote=not args.no_smote,
        target_mode="multiclass" if args.multiclass else "binary",
        random_state=args.seed,
    )
    pipeline = AegisPipeline(cfg)
    artifacts = pipeline.run()
    print("\n[aegis] artifacts written:")
    for k, v in artifacts.__dict__.items():
        print(f"  {k:>22}: {v}")


if __name__ == "__main__":
    main()
