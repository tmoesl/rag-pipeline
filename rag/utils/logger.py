"""
Simple logging setup for RAG pipeline.
Single instance, environment-configurable, suppresses noisy libraries.
"""

import logging
import sys
import time
from functools import lru_cache

from rag.config.settings import get_settings


@lru_cache
def setup_logger() -> logging.Logger:
    """
    Setup and return the RAG pipeline logger.
    Call once at startup, use the returned logger everywhere.
    """
    # Configure timestamp to UTC
    logging.Formatter.converter = time.gmtime

    # Get configuration from settings
    settings = get_settings()
    log_level = settings.log_level.upper()

    # Create main logger
    logger = logging.getLogger("rag")

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Set level
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Create console handler with clean format
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Suppress noisy libraries
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("psycopg").setLevel(logging.WARNING)

    logger.info("Logger initialized")
    return logger


# Create single logger instance
logger = setup_logger()
