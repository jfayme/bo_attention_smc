"""Constants loaded from the environment configuration.

The settings live in a TOML file so that every tunable parameter sits in one
place, outside the code that consumes it. Two file names are recognised:

- `.env.toml`, git-ignored, for machine-specific overrides;
- `settings.toml`, committed, holding the defaults.

The working directory is searched first so that a run can be pointed at its
own configuration, then the project root, so that imports keep working from
notebooks and test runners started elsewhere.

"""

import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent

SETTINGS_NAMES = (".env.toml", "settings.toml")


def _locate_settings() -> Path:
    """Find the settings file to load.

    Returns
    -------
    path
        Path to the first settings file found, searching the working
        directory before the project root.

    Raises
    ------
    RuntimeError
        If neither `.env.toml` nor `settings.toml` exists in either
        directory.

    """
    for directory in (Path.cwd(), PROJECT_ROOT):
        for name in SETTINGS_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate

    raise RuntimeError("Settings must be in .env.toml or settings.toml!")


# Load all settings.
SETTINGS_PATH = _locate_settings()

with SETTINGS_PATH.open("rb") as f:
    SETTINGS: dict[str, Any] = tomllib.load(f)
