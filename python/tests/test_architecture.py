"""Inner layers never import outer ones; the language never loads PyTorch."""

import ast
from pathlib import Path
import subprocess
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "lab"
ORDER = ["domain", "application", "infrastructure", "interfaces"]
# Modules at the package root, placed in the layer whose rules they follow.
ROOT_MODULES = {"dsl": "domain", "__init__": "domain", "__main__": "interfaces"}
THIRD_PARTY_ALLOWED = {"infrastructure", "interfaces"}


def layer_of(module):
    parts = module.split(".")
    if parts[0] != "lab":
        return None
    if len(parts) == 1:
        return "domain"
    return parts[1] if parts[1] in ORDER else ROOT_MODULES[parts[1]]


def imports(path):
    """Absolute names of every module a source file imports."""
    package = list(path.relative_to(PACKAGE.parent).parts[:-1])
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - node.level + 1]
                prefix = ".".join(base + ([node.module] if node.module else []))
                if node.module:
                    names.append(prefix)
                else:
                    names.extend(f"{prefix}.{alias.name}" for alias in node.names)
            else:
                names.append(node.module)
    return names


class ArchitectureTests(unittest.TestCase):
    def test_dependencies_point_inward(self):
        for path in PACKAGE.rglob("*.py"):
            relative = path.relative_to(PACKAGE).with_suffix("")
            own = relative.parts[0] if relative.parts[0] in ORDER else ROOT_MODULES[relative.parts[0]]
            for name in imports(path):
                target = layer_of(name)
                with self.subTest(module=str(relative), imports=name):
                    if target is not None:
                        self.assertLessEqual(ORDER.index(target), ORDER.index(own))
                    elif own not in THIRD_PARTY_ALLOWED:
                        self.assertIn(name.split(".")[0], sys.stdlib_module_names)

    def test_language_and_use_cases_load_without_torch(self):
        code = ("import sys, lab.dsl, lab.application.study, lab.application.report; "
                "assert not {'torch', 'numpy'} & set(sys.modules), sorted(sys.modules)")
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
