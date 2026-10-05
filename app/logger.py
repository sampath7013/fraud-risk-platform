import logging
import sys


# =========================================================
# Logging Configuration
# =========================================================

DEFAULT_LOGGER_NAME = "fraud-risk-platform"


def _configure_logger(name: str) -> logging.Logger:
    """
    Create and configure an application logger.

    The handler check prevents duplicate log messages when
    modules are imported repeatedly, including during
    Uvicorn --reload and pytest execution.
    """

    application_logger = logging.getLogger(name)

    application_logger.setLevel(logging.INFO)

    if not application_logger.handlers:
        console_handler = logging.StreamHandler(
            sys.stdout
        )

        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        )

        console_handler.setFormatter(formatter)

        application_logger.addHandler(
            console_handler
        )

    # Prevent the same message from also being emitted
    # through Python's root logger.
    application_logger.propagate = False

    return application_logger


def get_logger(
    name: str = DEFAULT_LOGGER_NAME,
) -> logging.Logger:
    """
    Return a configured logger.

    Existing modules such as risk_engine.py can continue:

        logger = get_logger(__name__)
    """

    return _configure_logger(name)


# =========================================================
# Default Application Logger
#
# Newer modules can simply use:
#
# from app.logger import logger
# =========================================================

logger = get_logger(DEFAULT_LOGGER_NAME)