"""Frozen, fingerprinted splits from independent deterministic seed streams."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

from .lookup import lookup_pair, mqar_example
from .oracles import validate_example
from .prefixes import prefix_pair, random_prefix
from .crasp import crasp_example
from .sequences import sequence_example
from .typed_dyck import random_typed, typed_pair
from .records import Example
from .specs import TaskSpec

GENERATOR_VERSION = 2


@dataclass(frozen=True)
class FrozenSplit:
    name: str
    spec: TaskSpec
    seed: int
    examples: tuple[Example, ...]

    @property
    def fingerprint(self):
        payload = {"version": GENERATOR_VERSION, "name": self.name,
                   "spec": self.spec.sampling_spec(), "seed": self.seed,
                   "examples": [asdict(example) for example in self.examples]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(json.dumps({
            "version": GENERATOR_VERSION, "name": self.name, "spec": self.spec.sampling_spec(),
            "seed": self.seed, "count": len(self.examples), "fingerprint": self.fingerprint,
        }, indent=2) + "\n")
        with (directory / "examples.jsonl").open("w") as stream:
            for example in self.examples:
                stream.write(json.dumps(asdict(example)) + "\n")


def build_split(spec, name, seed, count):
    if name not in ("train", "validation", "test") or seed < 0 or count < 1:
        raise ValueError("Need a valid split, nonnegative seed, and positive count")
    identity = json.dumps({"version": GENERATOR_VERSION, "spec": spec.problem_sampling_spec(),
                           "seed": seed, "split": name}, sort_keys=True)
    rng = random.Random(int.from_bytes(hashlib.sha256(identity.encode()).digest(), "big"))
    examples = []
    while len(examples) < count:
        length = spec.sample_length(rng)
        if spec.task == "random_lm":
            from .random_control import random_example

            # Single rows keep variable-length sampling IID, without paired lengths.
            example = random_example(spec, rng, length)
            validate_example(example, spec)
            examples.append(example)
            continue
        if spec.task == "mqar":
            pair = (mqar_example(spec, rng, length), mqar_example(spec, rng, length))
        elif spec.task == "lookup":
            pair = lookup_pair(spec, rng, length)
        elif spec.generative:
            pair = (sequence_example(spec, rng, length), sequence_example(spec, rng, length))
        elif spec.task == "crasp":
            pair = (crasp_example(spec, rng, length), crasp_example(spec, rng, length))
        elif spec.task == "dyck2":
            pair = ((random_typed(spec, rng, length), random_typed(spec, rng, length))
                    if rng.random() < 1 / 3 else typed_pair(spec, rng, length))
        elif rng.random() < 1 / 3:
            pair = (random_prefix(spec, rng, length), random_prefix(spec, rng, length))
        else:
            pair = prefix_pair(spec, rng, length)
        # Randomize pair order so row parity does not reveal the label.
        if rng.random() < 0.5:
            pair = pair[::-1]
        for example in pair:
            validate_example(example, spec)
            examples.append(example)
            if len(examples) == count:
                break
    return FrozenSplit(name, spec, seed, tuple(examples))
