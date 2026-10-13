"""Loading of the project configuration."""
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Read config.yaml into a plain dictionary."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
