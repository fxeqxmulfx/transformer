"""Validated task configurations with independent difficulty controls."""

from dataclasses import asdict, dataclass

from .vocabulary import IDENTITY_BASE
from .task_validation import validate_task

CORE_TASKS = ("mqar", "lookup", "dyck", "blocks")
RASP_TASKS = ("histogram", "histogram2", "mode", "most_freq", "copy", "reverse", "sort", "dyck2")
RASPL_TASKS = ("count", "mode", "copy", "sort", "addition", "parity", "boolean_and")
TASKS = tuple(dict.fromkeys((*CORE_TASKS, *RASP_TASKS, *RASPL_TASKS, "crasp")))
CONTROL_TASKS = ("random_lm",)
GENERATIVE_TASKS = tuple(dict.fromkeys((*RASP_TASKS[:-1], *RASPL_TASKS)))


@dataclass(frozen=True)
class TaskSpec:
    task: str = "lookup"
    length: int = 64
    min_length: int | None = None
    symbols: int = 32
    pairs: int = 8
    queries: int = 4
    hops: int = 2
    blocks: int = 3
    query_gap: int = 0
    alpha: float = 0.1
    neutral_fraction: float = 0.25
    max_neutral_gap: int | None = None
    max_balance: int = 8
    bracket_types: int = 2
    unique: bool = False
    histogram_bos: bool = True
    number_limit: int = 512
    scratchpad: str = "none"
    addition_order: str = "forward"
    index_hints: bool = False
    carry_sampling: str = "balanced"
    carry_length: int | None = None
    and_shift: bool = True
    and_region: str = "early"
    formula_depth: int = 2
    formula_seed: int = 0

    def __post_init__(self):
        if self.task not in (*TASKS, *CONTROL_TASKS):
            raise ValueError(f"Unknown task: {self.task}")
        minimum = self.min_length if self.min_length is not None else self.length
        if not 1 <= minimum <= self.length:
            raise ValueError("Lengths must satisfy 1 <= min_length <= length")
        validate_task(self, minimum)

    @property
    def family(self):
        if self.task in CONTROL_TASKS:
            return "memorization"
        if self.task in ("dyck", "blocks", "dyck2", "crasp"):
            return "prefix"
        return "rasp" if self.task in RASP_TASKS else "rasp_l" if self.task in RASPL_TASKS else self.task

    @property
    def generative(self):
        return self.task in (*GENERATIVE_TASKS, *CONTROL_TASKS)

    @property
    def number_base(self):
        return IDENTITY_BASE + self.symbols

    @property
    def uses_numbers(self):
        return (self.task in ("histogram", "histogram2", "count", "random_lm")
                or self.task == "mode" and self.scratchpad != "none"
                or self.task in ("addition", "parity") and self.index_hints)

    @property
    def vocab_size(self):
        if self.task == "mqar":
            return IDENTITY_BASE + 2 * self.symbols
        if self.task == "lookup":
            return IDENTITY_BASE + self.symbols
        if self.task == "dyck2":
            return IDENTITY_BASE + 2 * (self.bracket_types - 1)
        if self.generative:
            return self.number_base + self.number_limit + 1 if self.uses_numbers else self.number_base
        return IDENTITY_BASE

    @property
    def context_length(self):
        """Maximum serialized training/rollout context, including scratchpad."""
        n = self.length
        if self.task == "random_lm":
            return n + 3
        if self.task in ("copy", "reverse", "sort"):
            return 2 * n + 2
        if self.task in ("histogram", "histogram2", "most_freq"):
            bos = self.histogram_bos if self.task == "histogram" else True
            return 2 * n + (3 if bos else 1)
        if self.task == "mode":
            return n + 3 + (2 * min(n, self.symbols) if self.scratchpad != "none" else 0)
        if self.task == "count":
            return n + 4
        if self.task == "addition":
            return 6 * n + 9 if self.index_hints else 3 * n + 6
        if self.task == "parity":
            prompt = (2 if self.index_hints else 1) * n + 2
            output = (2 if self.index_hints else 1) * n + 1 if self.scratchpad != "none" else 1
            return prompt + output
        if self.task == "boolean_and":
            return n + 3
        return n

    def active_size(self, length):
        size = round((length - 1) * (1 - self.neutral_fraction))
        return size - size % 2 if self.task in ("dyck", "dyck2") else size

    def sample_length(self, rng):
        return rng.randint(self.min_length or self.length, self.length)

    def sampling_spec(self):
        """Only active task controls contribute to its random seed and fingerprint."""
        common = ("task", "length", "min_length")
        controls = {
            "mqar": ("symbols", "pairs", "queries", "query_gap", "alpha"),
            "lookup": ("symbols", "pairs", "queries", "query_gap", "hops"),
            "dyck": ("neutral_fraction", "max_neutral_gap", "max_balance"),
            "blocks": ("neutral_fraction", "max_neutral_gap", "blocks"),
            "dyck2": ("neutral_fraction", "max_neutral_gap", "max_balance", "bracket_types"),
            "histogram": ("symbols", "histogram_bos", "number_limit"),
            "histogram2": ("symbols", "number_limit"),
            "mode": ("symbols", "scratchpad"),
            "most_freq": ("symbols",),
            "copy": ("symbols", "unique"), "reverse": ("symbols", "unique"),
            "sort": ("symbols", "unique"),
            "count": ("symbols", "number_limit"),
            "addition": ("addition_order", "index_hints", "carry_sampling", "carry_length"),
            "parity": ("scratchpad", "index_hints"),
            "boolean_and": ("and_shift", "and_region") if self.and_shift else ("and_shift",),
            "crasp": ("formula_depth", "formula_seed"),
            "random_lm": ("symbols", "number_limit"),
        }
        extra = ("symbols", "number_limit") if self.uses_numbers else ()
        fields = asdict(self)
        return {name: fields[name] for name in (*common, *controls[self.task], *extra)}

    def problem_sampling_spec(self):
        """Response format controls share underlying problems for paired comparisons."""
        fields = self.sampling_spec()
        ignored = {"scratchpad", "addition_order", "index_hints", "histogram_bos"}
        if self.task != "count":
            ignored.add("number_limit")
        if self.task in ("addition", "parity"):
            ignored.add("symbols")
        return {name: value for name, value in fields.items() if name not in ignored}

    @property
    def run_name(self):
        """Variant names keep distinct experiments in distinct run directories."""
        if self.task == "histogram":
            return f"histogram-{'bos' if self.histogram_bos else 'no-bos'}"
        if self.task in ("copy", "reverse", "sort"):
            return f"{self.task}-{'unique' if self.unique else 'repeat'}"
        if self.task in ("mode", "parity"):
            hints = "-hints" if self.task == "parity" and self.index_hints else ""
            return f"{self.task}-{self.scratchpad}{hints}"
        if self.task == "addition":
            carry = self.carry_sampling if self.carry_length is None else f"carry-{self.carry_length}"
            return f"addition-{self.addition_order}-{'hints' if self.index_hints else 'plain'}-{carry}"
        if self.task == "boolean_and":
            return f"boolean-and-{'shift' if self.and_shift else 'random'}"
        if self.task == "dyck2":
            return f"dyck-{self.bracket_types}"
        if self.task == "crasp":
            return f"crasp-depth-{self.formula_depth}-program-{self.formula_seed}"
        return self.task
