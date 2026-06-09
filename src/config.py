"""
src/config.py
=============
Loads config.yaml and exposes its contents as a plain dict.

Usage
-----
    from src.config import load_config

    cfg = load_config()               # loads config.yaml from project root
    cfg = load_config("my_cfg.yaml")  # load a specific file

The returned dict matches the structure of config.yaml exactly, so you
can access values like:
    cfg["preprocessing"]["so_band"]
    cfg["paths"]["processed_dir"]

Why a module instead of just yaml.safe_load everywhere?
-  One place to add validation / defaults as the project grows.
-  Other modules import from here, so changing the YAML key name only
   requires a fix in this one file.
"""

import os
import yaml


# Path to the project root (two levels up from this file: src/ → root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


def load_config(config_path: str = None) -> dict:
    """
    Load the YAML configuration file and return it as a dict.

    Parameters
    ----------
    config_path : str, optional
        Absolute or relative path to a config YAML file.
        Defaults to config.yaml at the project root.

    Returns
    -------
    dict
        Full configuration dictionary matching config.yaml structure.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If a required top-level section is missing.
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Make sure config.yaml exists at the project root: {_PROJECT_ROOT}"
        )

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    _validate_config(cfg)
    _resolve_paths(cfg)

    return cfg


def _validate_config(cfg: dict) -> None:
    """
    Check that required top-level sections exist.

    Extend this function when you add new required sections to config.yaml.
    """
    required_sections = [
        "paths",
        "conditions",
        "preprocessing",
        "up_down",
        "firing_rates",
        "metrics",
        "statistics",
    ]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(
                f"Required section '{section}' is missing from config.yaml.\n"
                f"Please check the file against README.md."
            )


def _resolve_paths(cfg: dict) -> None:
    """
    Convert relative paths in cfg["paths"] to absolute paths
    (anchored at the project root), so callers never need to worry
    about the working directory.
    """
    for key, value in cfg["paths"].items():
        if value and not os.path.isabs(value):
            cfg["paths"][key] = os.path.join(_PROJECT_ROOT, value)


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT
