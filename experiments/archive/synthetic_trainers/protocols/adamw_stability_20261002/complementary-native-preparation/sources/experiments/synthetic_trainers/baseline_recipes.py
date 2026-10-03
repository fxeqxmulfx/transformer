"""Fixed AMSGradW/softmax recipes; freeze the plan before inspecting test scores."""

from dataclasses import dataclass, replace

from .config import ModelSpec, TrainConfig
from .specs import TASKS, TaskSpec
from .suite import expand_variants


@dataclass(frozen=True)
class Recipe:
    phase: str
    task: TaskSpec
    model: ModelSpec
    training: TrainConfig

    @property
    def name(self):
        return (f"{self.phase}/{self.task.run_name}/width-{self.model.width}"
                f"/noise-{self.training.label_noise:g}/seed-{self.training.seed}")


def recipes(phase="all", *, device="cuda", seeds=(0, 1, 2), suite_steps=1000,
            transition_steps=5000, capacity_steps=1000, control_steps=1000):
    """Equal update budgets within each phase, paired pools across model seeds.

    The suite is a one-seed calibration. Transition and capacity runs have three
    initialization seeds by default, with one fixed data seed (not population
    replicates). MQAR/lookup retain fixed association counts when length grows;
    their OOD probes measure spacing transfer. Other OOD probes grow problem size.
    """
    if phase not in ("all", "suite", "transitions", "capacity", "control"):
        raise ValueError("Unknown baseline phase")
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Model seeds must be distinct and nonnegative")
    model = ModelSpec(width=64, layers=2, heads=4, init_std=.02)
    config = TrainConfig(steps=suite_steps, batch_size=32, eval_every=100,
                         train_examples=128, validation_examples=32, test_examples=64,
                         learning_rate=.001, weight_decay=.1, grad_clip=None,
                         optimizer="amsgradw", beta1=.9, beta2=.999, optimizer_epsilon=1e-8,
                         device=device, data_seed=1, seed=seeds[0], study="double_descent",
                         eval_lengths=(32, 64), curve_tolerance=.02)
    selected = []
    if phase in ("all", "suite"):
        for task in TASKS:
            fields = dict(task=task, length=16, min_length=8, symbols=64,
                          pairs=4, queries=2, hops=2, number_limit=128)
            lengths = (32, 64)
            if task in ("mqar", "lookup"):
                fields.update(length=24, min_length=24)
                lengths = (48, 96)
            elif task in ("dyck", "dyck2"):
                fields.update(min_length=12)
            elif task == "addition":
                fields.update(length=4, min_length=2)
                lengths = (8, 16)
            spec = TaskSpec(**fields)
            for variant in expand_variants(spec):
                selected.append(Recipe("suite", variant, model, replace(config, eval_lengths=lengths)))
    if phase in ("all", "transitions", "capacity"):
        copy = TaskSpec(task="copy", length=8, min_length=4, symbols=8, number_limit=64)
        parity = TaskSpec(task="parity", length=8, symbols=8, number_limit=64)
        followup = replace(config, steps=transition_steps, eval_every=250,
                           train_examples=64, validation_examples=64, test_examples=128,
                           eval_lengths=(16, 32), split_policy="disjoint")
        if phase in ("all", "transitions"):
            for spec in (copy, parity, replace(parity, scratchpad="running")):
                for seed in seeds:
                    selected.append(Recipe("transitions", spec, model, replace(followup, seed=seed)))
        if phase in ("all", "capacity"):
            for width in (16, 32, 64, 128):
                for seed in seeds:
                    training = replace(followup, steps=capacity_steps, eval_every=100,
                                       label_noise=.2, noise_seed=2, seed=seed)
                    selected.append(Recipe("capacity", parity, replace(model, width=width), training))
    if phase in ("all", "control"):
        control = TaskSpec(task="random_lm", length=8, symbols=4, number_limit=64)
        for seed in seeds:
            training = replace(config, steps=control_steps, train_examples=8,
                               validation_examples=64, test_examples=128,
                               eval_lengths=(16, 32), seed=seed)
            selected.append(Recipe("control", control, model, training))
    return tuple(selected)
