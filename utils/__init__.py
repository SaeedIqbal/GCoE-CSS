# utils/__init__.py
"""
Initialization for the utils package.
"""
from .logging import (
    setup_logger,
    close_logger,
    log_config,
    log_metrics,
    log_info,
    log_warning,
    log_error
)

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "setup_logger",
    "close_logger",
    "log_config",
    "log_metrics",
    "log_info",
    "log_warning",
    "log_error"
]