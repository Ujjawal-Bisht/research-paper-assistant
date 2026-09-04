import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Directory where log files will be stored
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Parameters
    ----------
    name : str
        Usually __name__ of the calling module.

    Returns
    -------
    logging.Logger
        Configured application logger.
    """

    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers if get_logger()
    # is called multiple times for the same logger.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # General application log
    app_file_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    app_file_handler.setLevel(logging.DEBUG)
    app_file_handler.setFormatter(formatter)

    # Error log
    error_file_handler = RotatingFileHandler(
        LOG_DIR / "errors.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(app_file_handler)
    logger.addHandler(error_file_handler)

    return logger