#!/usr/bin/env python3
"""Fetch the arXiv source of every paper the repository cites into papers/.

    python3 scripts/papers.py [--dry-run]

A citation is a new-style arXiv identifier in a tracked file: `arXiv:<id>`,
an arxiv.org link, or a `papers/arXiv-<id>` path. Code vendored under
third_party/ cites for its own authors and is not read. A paper goes into
the folder the tree names for it, or else into `papers/arXiv-<id>/` with
`<id>` as cited, and at the version cited; a paper cited without a version
is fetched at its latest. A folder that is already there is left alone.

arXiv serves a submission's source as a gzipped tar archive, as a single
gzipped TeX file, or, when it has no source, as its PDF; each is unpacked
into the paper's folder. Requests are spaced three seconds apart, as arXiv
asks of scripts. papers/ is gitignored: the manuscripts are fetched, never
tracked. Only the standard library is used.
"""

import argparse
from collections import defaultdict
import gzip
import http.client
import io
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "papers"
NOT_READ = ("third_party/",)
IDENTIFIER = r"(\d{4}\.\d{4,5})(?!\d)(v\d+)?"
CITATION = re.compile(r"(?i:arxiv)(?::\s?|\s|\.org/(?:abs|pdf)/)" + IDENTIFIER)
FOLDER = re.compile(r"papers/(arXiv-" + IDENTIFIER + ")")
SOURCE = "https://arxiv.org/e-print/{}"
SERVED = re.compile(r"arXiv-(\d{4}\.\d{4,5}v\d+)")
AGENT = "transformer-papers/1.0 (scripts/papers.py)"
DELAY = 3.0


def tracked_files():
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    for name in listed.decode().split("\0"):
        path = ROOT / name
        if name and not name.startswith(NOT_READ) and path.is_file():
            yield path


def citations():
    """The versions cited of every paper ("" for none), and the folders the tree names for it."""
    versions, folders = defaultdict(set), defaultdict(set)
    for path in tracked_files():
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        text = data.decode(errors="replace")
        for identifier, version in CITATION.findall(text):
            versions[identifier].add(version)
        for folder, identifier, version in FOLDER.findall(text):
            versions[identifier].add(version)
            folders[identifier].add(folder)
    return versions, folders


def plan(versions, folders):
    """Every (folder, identifier to fetch) the citations ask for; no version fetches the latest."""
    targets = []
    for identifier in sorted(versions):
        cited = sorted(version for version in versions[identifier] if version)
        covered = set()
        for folder in sorted(folders[identifier]):
            version = folder.removeprefix(f"arXiv-{identifier}") or (cited[0] if len(cited) == 1 else "")
            covered.add(version)
            targets.append((folder, identifier + version))
        targets += [(f"arXiv-{identifier}{version}", identifier + version)
                    for version in cited if version not in covered]
        if not cited and not folders[identifier]:
            targets.append((f"arXiv-{identifier}", identifier))
    return targets


def present(folder):
    return folder.is_dir() and any(folder.iterdir())


def fetch(identifier):
    """The source of `identifier` as arXiv serves it, and the version served."""
    request = urllib.request.Request(SOURCE.format(identifier), headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        served = SERVED.search(response.headers.get("Content-Disposition", ""))
        return response.read(), served[1] if served else identifier


def unpack(data, folder):
    """Write a source archive, a single source file or a PDF into `folder`."""
    if data.startswith(b"%PDF"):
        (folder / "paper.pdf").write_bytes(data)
        return
    if not data.startswith(b"\x1f\x8b"):
        raise ValueError("arXiv served neither a gzip file nor a PDF")
    content = gzip.decompress(data)
    if tarfile.is_tarfile(io.BytesIO(content)):
        with tarfile.open(fileobj=io.BytesIO(content)) as archive:
            archive.extractall(folder, filter="data")
    elif content.startswith(b"%PDF"):
        (folder / "paper.pdf").write_bytes(content)
    else:
        (folder / "main.tex").write_bytes(content)


def download(folder, identifier):
    """Fetch `identifier` into papers/`folder`, which appears only once it is complete."""
    data, served = fetch(identifier)
    target, partial = PAPERS / folder, PAPERS / f".{folder}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        unpack(data, partial)
        if target.is_dir():
            target.rmdir()
        partial.rename(target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return served, len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="list the missing papers and fetch nothing")
    arguments = parser.parse_args()
    targets = plan(*citations())
    missing = [(folder, identifier) for folder, identifier in targets if not present(PAPERS / folder)]
    print(f"{len(targets)} papers cited, {len(targets) - len(missing)} in papers/, {len(missing)} missing", flush=True)
    failed = 0
    for number, (folder, identifier) in enumerate(missing):
        if arguments.dry_run:
            print(f"  missing  papers/{folder}  arXiv:{identifier}", flush=True)
            continue
        if number:
            time.sleep(DELAY)
        try:
            served, size = download(folder, identifier)
        except (OSError, EOFError, ValueError, http.client.HTTPException, tarfile.TarError, zlib.error) as error:
            failed += 1
            print(f"  failed   papers/{folder}  arXiv:{identifier}: {error}", flush=True)
        else:
            print(f"  fetched  papers/{folder}  arXiv:{served}, {size / 1e6:.1f} MB", flush=True)
    if failed:
        sys.exit(f"{failed} of {len(missing)} missing papers could not be fetched")


if __name__ == "__main__":
    main()
