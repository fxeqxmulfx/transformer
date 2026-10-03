"""Immutable blocks and the algebra that composes them.

Every word of the experiment language is a frozen dataclass deriving from
`Spec`. A spec only describes; builders in the infrastructure layer give it
behavior. Because specs are values, a configuration changes by producing a new
value: `swap` replaces the block at a dotted path, `substitute` replaces every
occurrence of a block, and `describe` renders the canonical JSON that
identifies a run.

A *kind* (`class Norm(Spec, kind=True)`) names a slot that alternative blocks
fill; it cannot be instantiated itself. Composite blocks check that each slot
holds a block of the right kind, so a wrong swap fails where it is written.
"""

from collections.abc import Iterator
from dataclasses import MISSING, dataclass, fields, replace
import hashlib
import json
import math

_BLOCKS: dict[str, type["Spec"]] = {}
_KINDS: set[type["Spec"]] = set()


@dataclass(frozen=True)
class Spec:
    """A block of the experiment language: an immutable description."""

    def __init_subclass__(cls, kind=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ in _BLOCKS:
            raise TypeError(f"Block name {cls.__name__} is already defined")
        _BLOCKS[cls.__name__] = cls
        if kind:
            _KINDS.add(cls)

    def __post_init__(self):
        if type(self) in _KINDS:
            raise TypeError(f"{type(self).__name__} is a kind of block; choose one of "
                            + ", ".join(block.__name__ for block in blocks_of(type(self))))
        self.check()

    def check(self):
        """Reject an invalid composition when it is written."""


def kinds():
    """Every slot of the language, in definition order."""
    return [cls for cls in _BLOCKS.values() if cls in _KINDS]


def blocks_of(kind):
    """The concrete blocks that fill a slot."""
    return [cls for cls in _BLOCKS.values() if issubclass(cls, kind) and cls not in _KINDS]


def composites():
    """Blocks that fill no slot: the fixed structure around the slots."""
    return [cls for cls in _BLOCKS.values()
            if cls not in _KINDS and not any(issubclass(cls, kind) for kind in _KINDS)]


def signature(cls):
    """`Name(field, field=default)` as an experiment file writes it."""
    parts = []
    for field in fields(cls):
        if field.default is not MISSING:
            parts.append(f"{field.name}={field.default!r}")
        else:
            parts.append(field.name)
    return f"{cls.__name__}({', '.join(parts)})"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_kind(value, kind, field):
    if not isinstance(value, kind):
        raise TypeError(f"{field} must be a {kind.__name__} block, got {value!r}")


def describe(value):
    """Canonical JSON form: every block names its type and all of its fields."""
    if isinstance(value, Spec):
        return {"type": type(value).__name__,
                **{field.name: describe(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, tuple):
        return [describe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Specs hold finite numbers only")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"{type(value).__name__} is not a spec value")


def fingerprint(value):
    """SHA-256 of the canonical description."""
    text = json.dumps(describe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def walk(value, path="") -> Iterator[tuple[str, Spec]]:
    """Every block with its dotted path, the root first."""
    if isinstance(value, Spec):
        yield path, value
        for field in fields(value):
            yield from walk(getattr(value, field.name), f"{path}.{field.name}" if path else field.name)
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from walk(item, f"{path}.{index}" if path else str(index))


def swap(value, path, new):
    """Replace the value at a dotted path; the rebuilt blocks are re-checked."""
    head, _, rest = path.partition(".")
    if isinstance(value, tuple):
        if not head.isdigit() or int(head) >= len(value):
            raise KeyError(f"No item {head!r} in a sequence of {len(value)}")
        index = int(head)
        return value[:index] + (swap(value[index], rest, new) if rest else new,) + value[index + 1:]
    if not isinstance(value, Spec) or head not in {field.name for field in fields(value)}:
        raise KeyError(f"{type(value).__name__} has no field {head!r}")
    return replace(value, **{head: swap(getattr(value, head), rest, new) if rest else new})


def substitute(value, old, new):
    """Replace every block equal to `old`, or of class `old`, by `new`."""
    if not (isinstance(old, Spec) or isinstance(old, type) and issubclass(old, Spec)):
        raise TypeError("Only blocks can be substituted; use swap for a single field")
    matches = (lambda item: isinstance(item, old)) if isinstance(old, type) else (lambda item: item == old)
    found = False

    def visit(item):
        nonlocal found
        if matches(item):
            found = True
            return new
        if isinstance(item, Spec):
            changes = {field.name: visit(getattr(item, field.name)) for field in fields(item)}
            if all(changes[name] is getattr(item, name) for name in changes):
                return item
            return replace(item, **changes)
        if isinstance(item, tuple):
            result = tuple(visit(element) for element in item)
            return item if all(a is b for a, b in zip(result, item)) else result
        return item

    result = visit(value)
    if not found:
        raise LookupError(f"No {old.__name__ if isinstance(old, type) else old!r} to substitute")
    return result
