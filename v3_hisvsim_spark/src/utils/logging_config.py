"""
Logging configuration for the quantum simulator.

Provides structured logging with appropriate log levels.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO).
        log_file: Optional file to write logs to.
        format_string: Optional custom format string.
        
    Returns:
        Configured logger instance.
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(message)s'
        )
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Root logger
    logger = logging.getLogger('quantum_simulator')
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__).
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(f'quantum_simulator.{name}')
