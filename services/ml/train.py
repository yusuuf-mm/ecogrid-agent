"""
services/ml/train.py

CLI entrypoint for training the solar forecast model.

Usage:
    python -m services.ml.train
    python -m services.ml.train --data-path data/raw/nrel_solar/722874_2022.csv
    python -m services.ml.train --data-path ... --output-path data/models/v2.joblib

Loads the DataFrame, trains the pipeline, prints evaluation metrics, and
saves the artifact with joblib. If the artifact directory does not exist
it is created.

Run from the repo root.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from loguru import logger

from services.ml.data_prep import load_and_prepare
from services.ml.model import evaluate, train

DEFAULT_OUTPUT_PATH = "data/models/solar_forecast.joblib"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the solar generation forecasting model."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Path to an NSRDB-derived CSV. If omitted, the synthetic "
            "data generator is used (good for development; not for "
            "production forecasts)."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for the trained artifact. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset held out for evaluation. Default: 0.2",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log output (loguru WARNING and above only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="WARNING" if args.quiet else "INFO")

    logger.info("Loading training data...")
    df = load_and_prepare(args.data_path)
    logger.info(f"Loaded {len(df)} rows from {args.data_path or 'synthetic generator'}.")

    logger.info("Training model...")
    pipeline = train(df, test_size=args.test_size)

    metrics = evaluate(pipeline, df)
    print("Final evaluation metrics (on full dataset):")
    print(f"  MAE  = {metrics['mae']:.3f} kW")
    print(f"  RMSE = {metrics['rmse']:.3f} kW")
    print(f"  R^2  = {metrics['r2']:.4f}")
    print(f"  n    = {metrics['n']}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info(f"Saved trained artifact to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
