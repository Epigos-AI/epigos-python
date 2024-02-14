import logging
import os

EPIGOS_DEBUG = str(os.getenv("EPIGOS_DEBUG") or False).lower() == "true"


def get_logger(name: str = "epigos-ai", verbose: bool = EPIGOS_DEBUG) -> logging.Logger:
    """
    Configure logging for the epigos-ai
    """
    level = logging.INFO if verbose else logging.ERROR
    # create logger
    log = logging.getLogger(name)
    log.setLevel(level)

    # create console handler and set level to debug
    handler = logging.StreamHandler()
    handler.setLevel(level)
    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # add formatter to handler
    handler.setFormatter(formatter)

    # add handler to logger
    log.addHandler(handler)
    log.propagate = False
    return log


logger = get_logger()
