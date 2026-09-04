from app.utils.logger import get_logger


def test_logger():
    logger = get_logger("test_logger")

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    logger.exception("Exception message")


if __name__ == "__main__":
    test_logger()