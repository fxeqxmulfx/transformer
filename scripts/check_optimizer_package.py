"""Exercise an installed optimizer wheel with lab imports explicitly blocked.

run.py package builds a copied source directory, installs its wheel into an
empty target, then runs this script under isolated Python. PyTorch is supplied
by the locked test environment; the wheel itself must declare only torch.
"""

import ast
from importlib import metadata
from pathlib import Path
import sys
import tempfile
import unittest


class BlockLab:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "lab":
            raise ImportError("The standalone optimizer package must not import lab")


def main():
    target, tests = map(lambda value: Path(value).resolve(), sys.argv[1:])
    sys.path.insert(0, str(target))
    sys.meta_path.insert(0, BlockLab())
    import lean_optim
    package = Path(lean_optim.__file__).resolve().parent
    assert package.is_relative_to(target), package
    distribution = next(metadata.distributions(path=[str(target)]))
    assert distribution.metadata["Name"] == "lean-optim"
    assert distribution.requires == ["torch>=2.0"], distribution.requires
    assert "ANSR" not in lean_optim.__all__
    allowed = sys.stdlib_module_names | {"torch", "lean_optim"}
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module]
            else:
                continue
            assert all(name.split(".")[0] in allowed for name in names), (path, names)
    assert (package / "LEAN_CORRESPONDENCE.md").is_file()
    with tempfile.TemporaryDirectory(prefix="lean-optim-cwd-") as cwd:
        import os
        os.chdir(cwd)
        suite = unittest.defaultTestLoader.discover(str(tests))
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert "lab" not in sys.modules
    if not result.wasSuccessful():
        sys.exit(1)
    print("Standalone wheel: torch-only metadata, no lab imports, installed sources, all tests passed")


if __name__ == "__main__":
    main()
