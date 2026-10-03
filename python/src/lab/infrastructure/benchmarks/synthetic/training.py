"""Training and teacher-forced evaluation on the splits of a synthetic benchmark.

Ports of `experiments/synthetic_trainers/training.py` (`train_run`) and
`metrics.py` (`masked_loss`, `Metrics`). The training loss is the mean cross
entropy over the supervised positions: written with `ignore_index` it has the
gradient of the historical `masked_loss`, which selected them first, and it
can be captured. Evaluation selects them through indices computed on the host
(`Rows`), so every metric adds the float32 values the historical evaluation
added, in its order, and its totals are bit-identical.
"""

import math

import torch
from torch.nn import functional as F

from ..samplers import EpochSampler
from . import generator
from .rows import Rows
from .splits import benchmark_splits
from .vocabulary import IGNORE

# Totals of an evaluation, before the per-class counts and correct predictions.
LOSS, EXAMPLE_LOSS, TARGETS, CORRECT, EXACT, TRANSITIONS, TRANSITION_CORRECT, CLASSES = range(8)


class SyntheticTask:
    """A synthetic benchmark's splits on one device, trained on with teacher forcing.

    A batch holds tokens, targets and changes on its second axis; the model
    reads the tokens and is supervised where a target is not IGNORE.
    """
    components = ("loss",)

    def __init__(self, spec, data_seed, device):
        if spec.task.generative:
            raise NotImplementedError("Free generation of a generated answer is not ported yet")
        if spec.study is not None:
            raise NotImplementedError("The observations of a memorization study are not ported yet")
        self.spec, self.device = spec, device
        splits, _ = benchmark_splits(spec, data_seed)
        self.generator = generator(spec.problem)
        self.vocab = max(split.generator.vocab for split in splits.values())
        self.fingerprints = {name: split.fingerprint for name, split in splits.items()}
        self.splits = {name: Rows(split.examples, device) for name, split in splits.items()}

    def sampler(self, batch, seed):
        """Shuffled epochs of the training rows, each ending with its remainder, as `train_run` drew them."""
        return EpochSampler(len(self.splits["train"]), batch, False, seed)

    def inputs(self, indices):
        return self.splits["train"].select(indices)

    def progress(self, seen):
        return {"examples_seen": seen, "epochs_seen": seen / len(self.splits["train"])}

    def accumulator(self):
        return torch.zeros(CLASSES + 2 * self.vocab, dtype=torch.float64, device=self.device)

    @staticmethod
    def forward(model, batch):
        """Logits at every position, and the targets."""
        return model(batch[:, 0]), batch[:, 1]

    @staticmethod
    def loss(output, targets):
        return F.cross_entropy(output.flatten(0, 1), targets.flatten(), ignore_index=IGNORE)

    @staticmethod
    def position_losses(output, targets):
        return SyntheticTask.loss(output, targets).reshape(1)

    @staticmethod
    def accumulate(model, chunk, sums):
        """Add a chunk's losses and counts to the float64 `sums`, as `Metrics.add` added them.

        Each example's loss is added position by position, as `scatter_add_`
        added it; the zero at an unsupervised position leaves it unchanged.
        """
        tokens, targets, changes = chunk.rows.unbind(1)
        logits = model(tokens)
        losses = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE, reduction="none")
        supervised = losses.index_select(0, chunk.supervised)
        valid, changes = targets != IGNORE, changes.bool()
        example = torch.zeros(len(targets), dtype=losses.dtype, device=losses.device)
        for column in losses.view_as(targets).unbind(1):
            example = example + column
        correct = logits.argmax(-1) == targets
        labels = targets.flatten().index_select(0, chunk.supervised)
        sums[:CLASSES].add_(torch.stack([
            supervised.mean().double() * len(chunk.supervised), (example / valid.sum(1)).sum().double(),
            valid.sum().double(), (correct & valid).sum().double(), (correct | ~valid).all(dim=1).sum().double(),
            changes.sum().double(), (correct & changes).sum().double()]))
        counts, right = sums[CLASSES:].view(2, -1).unbind(0)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=sums.dtype))
        right.index_add_(0, labels, correct.flatten().index_select(0, chunk.supervised).to(sums.dtype))

    def metrics(self, sums, examples):
        """`Metrics.report`, with a loss that is not finite reported as None, which no selection ranks."""
        totals, counts, right = sums[:CLASSES], sums[CLASSES:CLASSES + self.vocab], sums[CLASSES + self.vocab:]
        classes = {str(label): {"name": self.generator.token_name(label), "count": int(count),
                                "correct": int(correct), "accuracy": correct / count}
                   for label, (count, correct) in enumerate(zip(counts, right)) if count}
        loss = totals[LOSS] / totals[TARGETS]
        return {"loss": loss if math.isfinite(loss) else None, "example_loss": totals[EXAMPLE_LOSS] / examples,
                "token_accuracy": totals[CORRECT] / totals[TARGETS], "sequence_accuracy": totals[EXACT] / examples,
                "balanced_accuracy": sum(item["accuracy"] for item in classes.values()) / len(classes),
                "transition_accuracy": (totals[TRANSITION_CORRECT] / totals[TRANSITIONS]
                                        if totals[TRANSITIONS] else None),
                "target_count": int(totals[TARGETS]), "correct": int(totals[CORRECT]), "example_count": examples,
                "exact": int(totals[EXACT]), "transition_count": int(totals[TRANSITIONS]), "classes": classes}

    def analyze(self, history):
        """The splits' fingerprints, and the first observation whose validation `metric` reaches `target`."""
        spec = self.spec
        hit = None if spec.target is None else next(
            (row for row in history if row["validation"][spec.metric] >= spec.target), None)
        return {"split_fingerprints": self.fingerprints, "time_to_target": None if hit is None else {
            **{key: hit[key] for key in ("step", "training_seconds", "wall_seconds", "examples_seen")},
            "value": hit["validation"][spec.metric]}}
