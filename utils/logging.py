"""
Custom logging utilities for the GCoE-CSS framework.
Implements hierarchical logging with file, console, and WandB support.
"""

import logging
import os
import sys
from pathlib import Path
import datetime

# Optional: WandB integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logger(log_dir, rank=0, debug=False, use_wandb=False, project_name="gcoe-css"):
    """
    Set up a hierarchical logger for distributed training.

    :param log_dir: Directory to save log files.
    :param rank: Process rank (for distributed training). Only rank 0 logs to file.
    :param debug: If True, set log level to DEBUG. Else INFO.
    :param use_wandb: If True, integrate with Weights & Biases.
    :param project_name: Project name for WandB.
    :return: logging.Logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create unique log file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_rank{rank}_{timestamp}.log"

    # Create logger
    logger = logging.getLogger(f"gcoe_css.rank{rank}")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False  # Prevent double logging

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler (all ranks)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        fmt="[Rank %(process)d] %(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler (only rank 0)
    if rank == 0:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            fmt="[%(levelname)s @ %(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        # Initialize WandB (only once)
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=project_name,
                name=f"exp_{timestamp}",
                config={"log_dir": str(log_dir), "rank": rank}
            )
            logger.info(f"Weights & Biases initialized. Logging to project: {project_name}")

    logger.info(f"Logger initialized for rank {rank}. Log file: {log_file if rank == 0 else 'N/A'}")
    return logger


def close_logger(logger, rank=0):
    """
    Safely close logger and sync WandB.
    :param logger: Logger instance to close.
    :param rank: Process rank.
    """
    if rank == 0:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logger.info("Logger closed.")

    if rank == 0 and WANDB_AVAILABLE:
        wandb.finish()
        logger.info("Weights & Biases session finished.")


def log_config(logger, config, rank=0):
    """
    Log configuration dictionary.
    :param logger: Logger instance.
    :param config: Dict of configuration parameters.
    :param rank: Process rank.
    """
    if rank == 0:
        logger.info("Configuration:")
        for k, v in sorted(config.items()):
            logger.info(f"  {k}: {v}")


def log_metrics(logger, metrics, step=None, prefix="", rank=0):
    """
    Log metrics dictionary.
    :param logger: Logger instance.
    :param metrics: Dict of metrics (e.g., {'mIoU': 0.75, 'loss': 0.5}).
    :param step: Optional step/epoch number.
    :param prefix: Optional prefix (e.g., 'train/', 'val/').
    :param rank: Process rank.
    """
    if rank == 0:
        msg = f"{prefix}{'step' if step is None else 'epoch'} {step}: " if step is not None else f"{prefix}"
        msg += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(msg)

        # Log to WandB
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)


# Shortcut for basic logging
def log_info(logger, msg, rank=0):
    if rank == 0:
        logger.info(msg)


def log_warning(logger, msg, rank=0):
    if rank == 0:
        logger.warning(msg)


def log_error(logger, msg, rank=0):
    logger.error(msg)  # Log error on all ranks for visibility


__all__ = [
    'setup_logger',
    'close_logger',
    'log_config',
    'log_metrics',
    'log_info',
    'log_warning',
    'log_error'
]