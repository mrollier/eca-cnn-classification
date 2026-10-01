"""Shared command-line handling for the scripts in ``scripts/``.

Every script takes ``--config`` (a YAML file in ``configs/``) and any number of
``--set key=value`` overrides, e.g.::

    python scripts/train.py --config configs/trimmed_classes.yaml --set mode=faithful
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict

from .config import ROOT, Config, load_config, parse_overrides


def add_config_args(parser: argparse.ArgumentParser, default: str = "configs/base.yaml") -> None:
    parser.add_argument("--config", default=default, help=f"experiment YAML (default: {default})")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config field, e.g. --set mode=faithful (repeatable)")


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.config, parse_overrides(args.overrides))


def quiet_tensorflow() -> None:
    """Silence TensorFlow's start-up chatter (must run before importing it).

    ``TF_CPP_MIN_LOG_LEVEL`` silences the C++ side; the Python-side deprecation
    notices that Keras 2.15 prints on import go through the "tensorflow" logger.
    """
    import logging

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


def environment() -> Dict[str, Any]:
    """Versions and machine, recorded next to every result."""
    import numpy

    env = {"python": platform.python_version(), "numpy": numpy.__version__,
           "platform": platform.platform(), "processor": platform.processor() or platform.machine()}
    if "tensorflow" in sys.modules:
        import tensorflow as tf

        env["tensorflow"] = tf.__version__
        env["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    return env


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


RESULTS_DIR = ROOT / "data" / "results"
