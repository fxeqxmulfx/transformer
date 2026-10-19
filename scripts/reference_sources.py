#!/usr/bin/env python3
"""Check bundled historical sources, or rebuild them from their pinned commits.

Normal tests and checks read only files in this checkout. Rebuilding is an
explicit maintenance operation requiring the repository's own old Git objects.
The snapshots come from those commits, never from the implementation under test.
"""

import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = Path("python/tests/references/sources")
FROZEN = Path("scripts/fixtures/optimizer_benchmark")
BENCHMARK_REVISION = "2aec5b9"
ARCHIVED = "legacy/gpt_mini/src/infrastructure/benchmark/archive/experiments/optimizer_benchmark/"
METADATA = "experiments/archive/gpt_mini/optimizer_benchmark/results/rtx3050/metadata.json"


def revisions(root):
    """Read the existing immutable pins without importing the test environment."""
    tree = ast.parse((root / "python/tests/reference_fixtures.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "REVISIONS"
                                               for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing reference revision pins")


def archive_sources(root, revision):
    """Export the same source selection that the historical replay originally used."""
    rows = subprocess.check_output(["git", "ls-tree", "-r", revision, "--", "experiments", "legacy/gpt_mini"],
                                   cwd=root, text=True).splitlines()
    paths = [row.split("\t", 1)[1] for row in rows
             if row.startswith("120000 ") or row.endswith((".py", "reference-source.json", "/migration.json"))]
    paths = [path for path in paths
             if not any(part in path.split("/") for part in ("baselines", "protocols", "runs", "tests"))
             and not path.endswith("tinyshakespeare.txt")]
    if not paths:
        raise ValueError(f"No historical sources at {revision}")
    return subprocess.check_output(["git", "archive", revision, "--", *paths], cwd=root)


def archive_members(data):
    """Check that every archive member and link stays inside the exported tree."""
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = archive.getmembers()
    names = {member.name.rstrip("/") for member in members}
    for member in members:
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Nonlocal archive member: {member.name}")
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise ValueError(f"Unsupported archive member: {member.name}")
        if member.issym() or member.islnk():
            target = PurePosixPath(member.linkname)
            if target.is_absolute():
                raise ValueError(f"External archive link: {member.name}")
            resolved = posixpath.normpath(str(name.parent / target) if member.issym() else str(target))
            if resolved == ".." or resolved.startswith("../") or resolved not in names:
                raise ValueError(f"Missing or external archive target: {member.name}")
    return sum(not member.isdir() for member in members)


def rebuild(root=ROOT, output=None):
    """Bundle pinned independent sources and the measured benchmark's exact files."""
    output = root if output is None else output
    destination = output / BUNDLES
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"format": 1, "references": {}, "benchmark": {"revision": "", "files": {}}}
    for kind, revision in revisions(root).items():
        raw = archive_sources(root, revision)
        count = archive_members(raw)
        data = gzip.compress(raw, mtime=0)
        name = f"{revision}.tar.gz"
        (destination / name).write_bytes(data)
        manifest["references"][revision] = {"kind": kind, "archive": name,
                                            "sha256": hashlib.sha256(data).hexdigest(), "members": count}
    benchmark = manifest["benchmark"]
    benchmark["revision"] = subprocess.check_output(["git", "rev-parse", BENCHMARK_REVISION],
                                                    cwd=root, text=True).strip()
    metadata = json.loads((root / METADATA).read_text())
    for recorded, digest in metadata["protocol"]["source_hashes"].items():
        if not Path(recorded).is_absolute():
            continue
        prefix, separator, name = recorded.partition("/experiments/optimizer_benchmark/")
        if not prefix or not separator or ".." in PurePosixPath(name).parts:
            raise ValueError(f"Unknown benchmark source identifier: {recorded}")
        data = subprocess.check_output(["git", "show", f"{BENCHMARK_REVISION}:{ARCHIVED}{name}"], cwd=root)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Historical benchmark source hash differs: {name}")
        target = output / FROZEN / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        benchmark["files"][name] = digest
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    check(output)


def check(root=ROOT):
    """Verify the local artifacts without running Git, contacting a host, or reading another checkout."""
    destination = root / BUNDLES
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest["format"] != 1 or set(manifest["references"]) != set(revisions(root).values()):
        raise ValueError("Reference snapshot pins differ from the local manifest")
    for revision, entry in manifest["references"].items():
        if entry["archive"] != f"{revision}.tar.gz":
            raise ValueError(f"Unexpected snapshot path: {revision}")
        data = (destination / entry["archive"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Historical snapshot has changed: {revision}")
        if archive_members(data) != entry["members"]:
            raise ValueError(f"Historical snapshot member count differs: {revision}")
    for name, digest in manifest["benchmark"]["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"External benchmark source: {name}")
        if hashlib.sha256((root / FROZEN / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen benchmark source has changed: {name}")
    print(f"Verified {len(manifest['references'])} local reference archives and "
          f"{len(manifest['benchmark']['files'])} frozen benchmark sources")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="rebuild only from pinned historical Git objects")
    arguments = parser.parse_args()
    if arguments.rebuild:
        rebuild()
    else:
        check()


if __name__ == "__main__":
    main()
