"""Main training script - runs the complete ML pipeline.

Usage:
    python run_training.py                    # Train with auto-detected data
    python run_training.py --synthetic        # Train with synthetic data (for testing)
    python run_training.py --data path/to/data.csv
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.utils import generate_synthetic_data, load_raw_data, setup_logging
from src.preprocessing import (
    drop_identifier_columns,
    handle_missing_values,
    encode_categorical_columns,
    extract_target,
    split_data,
    normalize_features,
)
from src.feature_engineering import feature_engineering_pipeline
from src.train import training_pipeline, save_model, save_pipeline_artifacts
from src.evaluate import evaluate_model, save_evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Datathon Educacao - Passos Magicos - Pipeline de Treinamento")
    parser.add_argument("--data", type=str, help="Path to the dataset CSV/Excel file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--model", type=str, default=None, help="Model name (LogisticRegression, RandomForest, GradientBoosting, XGBoost, SVM)")
    parser.add_argument("--scoring", type=str, default="f1", help="Scoring metric (f1, recall, precision, accuracy)")
    parser.add_argument("--samples", type=int, default=500, help="Number of synthetic samples")

    args = parser.parse_args()
    setup_logging()

    logger.info("=" * 60)
    logger.info("DATATHON EDUCACAO - PASSOS MAGICOS - ML PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load data
    if args.synthetic:
        logger.info("Using synthetic data for training")
        df = generate_synthetic_data(n_samples=args.samples)
    elif args.data:
        df = load_raw_data(args.data)
    else:
        df = load_raw_data()

    logger.info(f"Dataset shape: {df.shape}")

    # Step 2: Drop identifiers
    df = drop_identifier_columns(df)

    # Step 3: Handle missing values
    df = handle_missing_values(df, strategy="median")

    # Step 4: Feature engineering (before target extraction)
    df = feature_engineering_pipeline(df)

    # Step 5: Extract target
    X, y = extract_target(df)

    # Step 6: Encode categorical columns
    X, encoders = encode_categorical_columns(X)

    # Step 7: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 8: Normalize
    X_train, X_test, scaler = normalize_features(X_train, X_test)

    # Prepare artifacts
    artifacts = {
        "encoders": encoders,
        "scaler": scaler,
        "feature_names": X_train.columns.tolist(),
    }

    # Step 9: Train
    model, cv_results, best_params = training_pipeline(
        X_train, y_train, artifacts,
        model_name=args.model,
        scoring=args.scoring,
    )

    # Step 10: Evaluate
    results = evaluate_model(model, X_test, y_test, feature_names=X_train.columns.tolist())
    save_evaluation_report(results)

    # Log to monitoring
    from src.monitoring import log_model_metrics
    log_model_metrics(results["metrics"], type(model).__name__)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Model: {type(model).__name__}")
    logger.info(f"Best params: {best_params}")
    logger.info(f"Test metrics: {results['metrics']}")
    logger.info("=" * 60)

    return model, results


if __name__ == "__main__":
    main()
