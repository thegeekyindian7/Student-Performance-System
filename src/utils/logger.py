import logging
from pathlib import Path


def setup_logger(name: str = "pipeline") -> logging.Logger:
    log_dir = Path("artifacts/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger  

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_dir / "pipeline.log")
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
