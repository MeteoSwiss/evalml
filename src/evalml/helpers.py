import logging


def setup_logger(logger_name, log_file, level=logging.INFO):
    """
    Setup a logger with the specified name and log file path.

    Can be used to set up loggers from python scripts `run` directives
    used in the Snakemake workflow.

    Parameters
    ----------
    logger_name : str
        The name of the logger.
    log_file : str
        The file path where the log messages will be written.
    level : int, optional
        The logging level (default is logging.INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
