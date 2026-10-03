"""Read an experiment: a folder whose `experiment.py` defines its `experiments` mapping."""

from pathlib import Path
import runpy

from ..application.study import Study

FILE = "experiment.py"


def folder(path):
    """The experiment folder `path` names: the folder itself, or the folder of its `experiment.py`."""
    path = Path(path)
    root = path.parent if path.name == FILE else path
    if not (root / FILE).is_file():
        raise FileNotFoundError(f"{path} is not an experiment folder: it holds no {FILE}")
    return root.resolve()


def load(path):
    """The study an experiment folder defines, named after the folder."""
    root = folder(path)
    namespace = runpy.run_path(str(root / FILE), run_name=f"experiment_{root.name}")
    if "experiments" not in namespace:
        raise LookupError(f"{root / FILE} defines no `experiments` mapping")
    return Study(root.name, (root / FILE).read_text(), namespace["experiments"])


def runs_root(path):
    """Where an experiment's runs live: `runs/` in its folder, a directory per label."""
    return folder(path) / "runs"
