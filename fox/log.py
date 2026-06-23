import logging
from pathlib import Path


def set_up(output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # terminal output
            logging.FileHandler(output_dir / Path("app.log")),  # file output
        ],
    )


def start(msg: str) -> None:
    logging.info(f"Starting {msg}...")


def end(msg: str) -> None:
    logging.info(f"Finished {msg}.")


def log_cwd() -> None:
    logging.info(f"Current working directory: {Path.cwd()}")
