"""Run an archived implementation in a separate process, with no lab imports."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import torch

def package(name, path):
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py",
                                                submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def main():
    kind, source, fixture, output, corpus, selection = sys.argv[1:]
    source = Path(source)
    golden = json.loads(Path(fixture).read_text())
    if selection not in ("*", "optimizer_steps"):
        golden["runs"] = {selection: golden["runs"][selection]}
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):
        if kind == "modular":
            from modular import replay
            replay(golden)
        elif kind == "recall":
            sys.path.insert(0, str(source / "experiments/convex_mqar/src"))
            from recall import replay
            replay(golden)
        elif kind in ("synthetic", "memorization"):
            from synthetic import replay
            replay(golden)
        else:
            package("gpt_mini", source / "legacy/gpt_mini/src")
            if selection == "optimizer_steps":
                from optimizer_steps import replay
                replay(golden)
            else:
                from text import replay
                replay(golden, corpus)
    if any(name == "lab" or name.startswith("lab.") for name in sys.modules):
        raise RuntimeError("An independent reference imported the implementation under test")
    golden["runtime_reference"] = {"torch": torch.__version__, "threads": torch.get_num_threads()}
    Path(output).write_text(json.dumps(golden))


if __name__ == "__main__":
    main()
