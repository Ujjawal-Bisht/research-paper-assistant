import logging
from logging.handlers import RotatingFileHandler

from app.config.config import (
    LOG_BACKUP_COUNT,
    LOG_DIR,
    LOG_FORMAT,
    LOG_MAX_BYTES,
)


LOG_DIR.mkdir(parents=True, exist_ok=True)


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
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    app_file_handler.setLevel(logging.DEBUG)
    app_file_handler.setFormatter(formatter)

    # Error log
    error_file_handler = RotatingFileHandler(
        LOG_DIR / "errors.log",
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(app_file_handler)
    logger.addHandler(error_file_handler)

    return logger