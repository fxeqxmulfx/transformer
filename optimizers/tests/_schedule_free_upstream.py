"""Load the unmodified published optimizer from its hash-pinned source archive."""

from __future__ import annotations

import hashlib
import tarfile
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@lru_cache(maxsize=1)
def upstream_adamw() -> type[torch.optim.Optimizer]:
    """Read only the verified 1.4.1 source; require neither network nor Git."""
    archive = Path(__file__).parent / "fixtures" / "schedulefree-1.4.1.tar.gz"
    digest = "69ef25601d1fc0d8dd00cb36f9af78833f88b7846f1bb6ddecc9f144f3e9f7cb"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        msg = "Published schedulefree 1.4.1 archive hash differs"
        raise ValueError(msg)
    name = "schedulefree-1.4.1/schedulefree/adamw_schedulefree.py"
    with tarfile.open(archive) as bundle:
        source = bundle.extractfile(name)
        if source is None:
            msg = "Published AdamW source is missing"
            raise ValueError(msg)
        code = compile(source.read(), name, "exec")
    module = ModuleType("_schedulefree_141")
    exec(code, module.__dict__)  # noqa: S102 - Unmodified source verified by the pinned SHA256 above.
    return module.AdamWScheduleFree
