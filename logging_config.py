"""Central logging configuration for the whole project.

Modules do not configure logging themselves; they ask for a named logger with
`get_logger(__name__)` and emit records through it. Handlers, levels and the
record format are attached once, by `setup_logging`, from the `[logger]` table
of the settings file. Keeping the names per-module while the configuration is
shared means a single call decides where the records go, and every record
still says which module wrote it.

Entry points (scripts, notebooks, the test runner) call `setup_logging` once
at start-up. Library code never does: importing a module must not decide where
someone else's log records end up.

"""

import logging
from pathlib import Path

from settings import SETTINGS

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s (%(name)s)"
DATE_FORMAT = "%d-%b-%y %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return the logger a module should write to.

    Parameters
    ----------
    name
        Name of the logger, conventionally the module's `__name__`.

    Returns
    -------
    logger
        The named logger, which inherits the handlers and the level set by
        `setup_logging`.

    """
    return logging.getLogger(name)


def setup_logging() -> None:
    """Attach the project's handlers to the root logger.

    The `[logger]` table of the settings file sets the lowest severity
    recorded, the file the records are appended to (an empty path writes
    none) and whether they are also written to stderr. Warnings issued
    through the `warnings` module are captured, so that they reach the same
    handlers.

    Notes
    -----
    Any handler already on the root logger is removed first, so calling
    this twice leaves the same handlers as calling it once.

    """
    config = SETTINGS["logger"]
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    handlers: list[logging.Handler] = []
    if config["path"]:
        path = Path(config["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="a"))
    if config["console"]:
        handlers.append(logging.StreamHandler())

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(config["level"])
    logging.captureWarnings(True)
