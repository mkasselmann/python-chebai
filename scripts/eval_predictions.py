#!/usr/bin/env python
"""
Script to evaluate model predictions on the test set.
Executes notebook cells 5 and 6 and saves results.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse

import torch
from chebai.result.prediction import Predictor

# ============================================================
# CONFIGURATION - Edit these paths before running
# ============================================================

# Path to model checkpoint (REQUIRED)
CHECKPOINT_PATH = "logs/wandb/run-20261111_021924-pst7wzsa/files/checkpoints/best_micro_f1_epoch=151_val_loss=0.0110_val_macro-f1=0.6575_val_micro-f1=0.8969.ckpt"  # e.g., "/scratch/user/run-123/checkpoint.ckpt"

# Project directory (optional, auto-detected if None)
PROJECT_DIR = None  # e.g., "/home/user/python-chebai"

# Path to processed run directory with splits.csv (optional)
PROCESSED_DIR = "data/chebi_v244/ChEBI50/processed/run_20261111_021403_48716b98e8e443889ca11bac75d0cc80"  # e.g., "/data/processed/run-456"

# Output directory for results
OUTPUT_DIR = "results"  # e.g., "/scratch/user/results"

# Save results to disk (set to False to only print metrics)
SAVE_RESULTS = True

# ============================================================
# END CONFIGURATION
# ============================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_device():
    """Setup device (GPU or CPU)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def load_predictor(checkpoint_path: str) -> Predictor:
    """Load predictor from checkpoint."""
    logger.info(f"Loading predictor from: {checkpoint_path}")
    predictor = Predictor(checkpoint_path=checkpoint_path, compile_model=False)
    logger.info(f"Loaded datamodule: {type(predictor._dm).__name__}")
    logger.info(
        f"Input dim: {predictor._dm.input_dim}, Output dim: {predictor._dm.output_dim}"
    )
    return predictor


def evaluate_predictions(predictor: Predictor) -> tuple:
    """
    Cell 5: Predict the whole test set.
    Returns predictions and labels as tensors.
    """
    logger.info("Starting evaluation of test set...")
    test_dl = predictor._dm.test_dataloader()

    preds = []
    labels = []

    total_batches = len(test_dl)
    logger.info(f"Processing {total_batches} batches...")

    for batch_idx, batch in enumerate(test_dl):
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Processed batch {batch_idx + 1}/{total_batches}")

        batch_preds = predictor._model.test_step(batch, batch_idx)
        preds.append(batch_preds["preds"])
        labels.append(batch_preds["labels"])

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    logger.info(f"Predictions shape: {preds.shape}")
    logger.info(f"Labels shape: {labels.shape}")

    return preds, labels


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Cell 6: Compute evaluation metrics.
    """
    logger.info("Computing evaluation metrics...")

    from torchmetrics import F1Score, Recall, Precision

    metrics = {}

    # Micro metrics
    micro_f1 = F1Score(num_labels=preds.shape[1], task="multilabel", average="micro")
    micro_precision = Precision(
        num_labels=preds.shape[1], task="multilabel", average="micro"
    )
    micro_recall = Recall(num_labels=preds.shape[1], task="multilabel", average="micro")

    metrics["micro_f1"] = float(micro_f1(preds, labels))
    metrics["micro_precision"] = float(micro_precision(preds, labels))
    metrics["micro_recall"] = float(micro_recall(preds, labels))

    # Macro metrics
    macro_f1 = F1Score(num_labels=preds.shape[1], task="multilabel", average="macro")
    macro_precision = Precision(
        num_labels=preds.shape[1], task="multilabel", average="macro"
    )
    macro_recall = Recall(num_labels=preds.shape[1], task="multilabel", average="macro")

    metrics["macro_f1"] = float(macro_f1(preds, labels))
    metrics["macro_precision"] = float(macro_precision(preds, labels))
    metrics["macro_recall"] = float(macro_recall(preds, labels))

    return metrics


def save_results(
    preds: torch.Tensor, labels: torch.Tensor, metrics: dict, output_dir: str
):
    """Save predictions, labels, and metrics to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = output_dir / f"eval_results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to: {results_dir}")

    # Save predictions and labels as PyTorch tensors
    torch.save(preds, results_dir / "predictions.pt")
    torch.save(labels, results_dir / "labels.pt")
    logger.info("✓ Saved predictions.pt and labels.pt")

    # Save predictions and labels as numpy arrays
    import numpy as np

    np.save(results_dir / "predictions.npy", preds.cpu().numpy())
    np.save(results_dir / "labels.npy", labels.cpu().numpy())
    logger.info("✓ Saved predictions.npy and labels.npy")

    # Save metrics as JSON
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("✓ Saved metrics.json")

    # Save a summary report
    summary = {
        "timestamp": timestamp,
        "predictions_shape": str(preds.shape),
        "labels_shape": str(labels.shape),
        "metrics": metrics,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("✓ Saved summary.json")

    # Print metrics to console
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Micro F1:        {metrics['micro_f1']:.6f}")
    logger.info(f"Micro Precision: {metrics['micro_precision']:.6f}")
    logger.info(f"Micro Recall:    {metrics['micro_recall']:.6f}")
    logger.info(f"Macro F1:        {metrics['macro_f1']:.6f}")
    logger.info(f"Macro Precision: {metrics['macro_precision']:.6f}")
    logger.info(f"Macro Recall:    {metrics['macro_recall']:.6f}")
    logger.info("=" * 60)
    logger.info(f"Results saved in: {results_dir}\n")

    return results_dir


def main():
    """Main execution function."""
    # Use config variables if set, otherwise fall back to command line arguments
    checkpoint = CHECKPOINT_PATH
    project_dir = PROJECT_DIR
    processed_dir = PROCESSED_DIR
    output_dir = OUTPUT_DIR
    save_results = SAVE_RESULTS

    # Parse command line arguments (these override config if provided)
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration:
  Edit CHECKPOINT_PATH, PROJECT_DIR, PROCESSED_DIR, OUTPUT_DIR at the top of this script.
  Or override with command line arguments (command line takes precedence).

Examples:
  # Use configuration from top of script
  python scripts/eval_predictions.py

  # Override with command line arguments
  python scripts/eval_predictions.py \\
    --checkpoint /scratch/user/run-123/checkpoint.ckpt \\
    --project-dir /home/user/python-chebai
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides CHECKPOINT_PATH)",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Project directory (overrides PROJECT_DIR)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Path to processed run directory (overrides PROCESSED_DIR)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides OUTPUT_DIR)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results (only print metrics)",
    )

    args = parser.parse_args()

    # Override config with command line arguments if provided
    if args.checkpoint:
        checkpoint = args.checkpoint
    if args.project_dir:
        project_dir = args.project_dir
    if args.processed_dir:
        processed_dir = args.processed_dir
    if args.output_dir:
        output_dir = args.output_dir
    if args.no_save:
        save_results = False

    # Validate checkpoint is provided
    if not checkpoint:
        parser.error(
            "CHECKPOINT_PATH is required!\n"
            "  Option 1: Set CHECKPOINT_PATH at the top of this script\n"
            "  Option 2: Provide --checkpoint /path/to/checkpoint.ckpt"
        )

    # Determine project directory
    if project_dir:
        project_dir = Path(project_dir)
    else:
        project_dir = Path(__file__).parent.parent

    # Change to project directory
    os.chdir(project_dir)
    logger.info(f"Working directory: {project_dir}")

    # Setup
    device = setup_device()

    # Load predictor
    predictor = load_predictor(checkpoint)

    # Cell 5: Evaluate predictions
    preds, labels = evaluate_predictions(predictor)

    # Cell 6: Compute metrics
    metrics = compute_metrics(preds, labels)

    # Save results
    if save_results:
        save_results(preds, labels, metrics, output_dir)
    else:
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS (NOT SAVED)")
        logger.info("=" * 60)
        logger.info(f"Micro F1:        {metrics['micro_f1']:.6f}")
        logger.info(f"Micro Precision: {metrics['micro_precision']:.6f}")
        logger.info(f"Micro Recall:    {metrics['micro_recall']:.6f}")
        logger.info(f"Macro F1:        {metrics['macro_f1']:.6f}")
        logger.info(f"Macro Precision: {metrics['macro_precision']:.6f}")
        logger.info(f"Macro Recall:    {metrics['macro_recall']:.6f}")
        logger.info("=" * 60 + "\n")

    logger.info("✓ Evaluation complete!")


if __name__ == "__main__":
    main()
