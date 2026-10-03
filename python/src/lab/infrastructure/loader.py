"""Read an experiment file: execute it and take its `experiments` mapping."""

from pathlib import Path
import runpy

from ..application.study import Study


def load(path):
    """The study an experiment file defines; runs live beside it in `runs/<stem>/`."""
    path = Path(path)
    if path.suffix != ".py" or not path.is_file():
        raise FileNotFoundError(f"{path} is not an experiment file")
    namespace = runpy.run_path(str(path), run_name=f"experiment_{path.stem}")
    if "experiments" not in namespace:
        raise LookupError(f"{path} defines no `experiments` mapping")
    return Study(path.stem, path.read_text(), namespace["experiments"])


def runs_root(path):
    return Path(path).resolve().parent / "runs"
