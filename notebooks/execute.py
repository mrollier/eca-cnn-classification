"""Execute ``reproduce_chapter.ipynb`` headlessly and save it with its outputs.

    python notebooks/execute.py

Needs the ``notebook`` extra (``pip install -e '.[notebook]'``); without it the
script says so and exits successfully, so ``reproduce.py all`` still passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        print("nbformat/nbclient not installed (pip install -e '.[notebook]'); skipping")
        return 0
    path = HERE / "reproduce_chapter.ipynb"
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE.parent)}}).execute()
    nbformat.write(nb, path)
    print(f"executed {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
