"""
Logging configuration for the project.

This module provides a centralized configuration for the logging system.
It defines different log levels, formatters, and handlers for various parts
of the application.

Usage:
    Import this module and call setup_logging() early in your application
    to configure the logging system:

    ```python
    from logging_config import setup_logging
    setup_logging()
    ```
"""

import os
import sys
import logging
import logging.config
from pathlib import Path


def setup_logging(default_level=logging.INFO, log_dir="logs"):
    """
    Setup logging configuration for the project.
    
    Parameters
    ----------
    default_level : int, optional
        Default logging level. Default is logging.INFO.
    log_dir : str, optional
        Directory to store log files. Default is "logs".
    
    Returns
    -------
    None
        The function configures the logging system directly.
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Define logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': log_path / 'simulation.log',
                'maxBytes': 10485760,  # 10 MB
                'backupCount': 5,
                'encoding': 'utf8'
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': log_path / 'error.log',
                'maxBytes': 10485760,  # 10 MB
                'backupCount': 5,
                'encoding': 'utf8'
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file', 'error_file'],
                'level': default_level,
                'propagate': True
            },
            'src.dynamics': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': False
            },
            'src.dynamics.manifolds': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': False
            },
            'src.utils': {
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': False
            },
        }
    }
    
    # Apply the configuration
    logging.config.dictConfig(config)
    
    # Log that logging has been configured
    logger = logging.getLogger(__name__)
    logger.debug("Logging configuration applied")


# Setup logging when the module is imported (if not imported elsewhere)
if __name__ == "__main__":
    setup_logging() 