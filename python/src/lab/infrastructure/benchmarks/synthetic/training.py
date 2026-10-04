"""Training and evaluation on the splits of a synthetic benchmark.

Ports of `experiments/synthetic_trainers/training.py` (`train_run`) and
`metrics.py` (`masked_loss`, `Metrics`, `evaluate`). The training loss is the
mean cross entropy over the supervised positions: written with `ignore_index`
it has the gradient of the historical `masked_loss`, which selected them
first, and it can be captured. Evaluation selects them through indices
computed on the host (`Rows`), so every metric adds the float32 values the
historical evaluation added, in its order, and its totals are bit-identical.
A split of generated answers is also scored on free generation
(`generation`), as `evaluate` scored it; a training split, by teacher forcing
alone. A memorization study measures more (`memorization`).
"""

import math

import torch
from torch.nn import functional as F

from ..samplers import EpochSampler
from . import generator
from .generation import TOTALS, rollout, score, static_rollout
from .memorization import Measures
from .rows import Rows
from .splits import benchmark_splits
from .vocabulary import IGNORE

# Totals of teacher forcing, then those of free generation from GENERATED, before the count of each label, its
# correct predictions, and its correct generated tokens.
LOSS, EXAMPLE_LOSS, TARGETS, CORRECT, EXACT, TRANSITIONS, TRANSITION_CORRECT, GENERATED = range(8)
CLASSES = GENERATED + TOTALS


class SyntheticTask:
    """A synthetic benchmark's splits on one device, trained on with teacher forcing.

    A batch holds tokens, targets and changes on its second axis; the model
    reads the tokens and is supervised where a target is not IGNORE.
    """
    components = ("loss",)

    def __init__(self, spec, data_seed, device):
        self.spec, self.device = spec, device
        splits, noise = benchmark_splits(spec, data_seed)
        self.generator = generator(spec.problem)
        self.vocab = max(split.generator.vocab for split in splits.values())
        self.fingerprints = {name: split.fingerprint for name, split in splits.items()}
        self.splits = {}
        for name, split in splits.items():
            # Labels no noise changed are the observed ones, and their rows are measured once.
            same = name == "train_clean" and split.examples == splits["train"].examples
            self.splits[name] = self.splits["train"] if same else Rows(split.examples, device,
                                                                       generate=not name.startswith("train"))
        self.measures = None if spec.study is None else Measures(spec, splits, noise, self.splits, device)

    def sampler(self, batch, seed):
        """Shuffled epochs of the training rows, each ending with its remainder, as `train_run` drew them."""
        return EpochSampler(len(self.splits["train"]), batch, False, seed)

    def inputs(self, indices, static=False):
        return self.splits["train"].select(indices, static)

    def progress(self, seen):
        return {"examples_seen": seen, "epochs_seen": seen / len(self.splits["train"])}

    def accumulator(self):
        return torch.zeros(CLASSES + 3 * self.vocab, dtype=torch.float64, device=self.device)

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

    def accumulate(self, model, chunk, sums, static):
        """Add a chunk's losses and counts to the float64 `sums`, as `Metrics.add` added them.

        Each example's loss is added position by position, as `scatter_add_`
        added it; the zero at an unsupervised position leaves it unchanged.
        Generated answers are generated from their prompts, by
        `static_rollout` when `static` and otherwise by `rollout`, and scored
        as `GenerationMetrics.add` scored them; their tokens are the
        supervised targets, whose counts they share.
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
        sums[:GENERATED].add_(torch.stack([
            supervised.mean().double() * len(chunk.supervised), (example / valid.sum(1)).sum().double(),
            valid.sum().double(), (correct & valid).sum().double(), (correct | ~valid).all(dim=1).sum().double(),
            changes.sum().double(), (correct & changes).sum().double()]))
        counts, right, written = sums[CLASSES:].view(3, -1).unbind(0)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=sums.dtype))
        right.index_add_(0, labels, correct.flatten().index_select(0, chunk.supervised).to(sums.dtype))
        if chunk.answers is not None:
            generate = static_rollout if static else rollout
            totals, generated = score(chunk.answers, *generate(model, chunk.answers), self.spec.task.final_token)
            sums[GENERATED:CLASSES].add_(totals)
            written.index_add_(0, chunk.answers.answers.clamp(min=0).flatten(), generated.flatten().to(sums.dtype))

    def metrics(self, sums, rows):
        """`Metrics.report`, or for generated answers the report of `evaluate`.

        That is the report of `GenerationMetrics`, with the loss and example
        loss of teacher forcing, and its whole report as `teacher_forced`. A
        loss that is not finite is reported as None, which no selection ranks.
        """
        totals, vocab, examples = sums[:CLASSES], self.vocab, len(rows)
        counts, right, written = (sums[CLASSES + kind * vocab:CLASSES + (kind + 1) * vocab] for kind in range(3))
        loss = totals[LOSS] / totals[TARGETS]
        teacher = {"loss": loss if math.isfinite(loss) else None, "example_loss": totals[EXAMPLE_LOSS] / examples,
                   **self.accuracy(totals, counts, examples, totals[CORRECT], totals[EXACT],
                                   totals[TRANSITION_CORRECT], right)}
        if not rows.generates:
            return teacher
        correct, exact, final, ended, generated, extra, transition_correct = totals[GENERATED:CLASSES]
        return {**self.accuracy(totals, counts, examples, correct, exact, transition_correct, written),
                "final_answer_accuracy": final / examples, "eos_rate": ended / examples,
                "generated_tokens": int(generated), "extra_tokens": int(extra), "loss": teacher["loss"],
                "example_loss": teacher["example_loss"], "teacher_forced": teacher}

    def accuracy(self, totals, counts, examples, correct, exact, transition_correct, right):
        """The accuracies of predictions with `correct` tokens, `exact` examples, and `right` tokens of each label."""
        classes = {str(label): {"name": self.generator.token_name(label), "count": int(count),
                                "correct": int(hit), "accuracy": hit / count}
                   for label, (count, hit) in enumerate(zip(counts, right)) if count}
        return {"token_accuracy": correct / totals[TARGETS], "sequence_accuracy": exact / examples,
                "balanced_accuracy": sum(item["accuracy"] for item in classes.values()) / len(classes),
                "transition_accuracy": transition_correct / totals[TRANSITIONS] if totals[TRANSITIONS] else None,
                "target_count": int(totals[TARGETS]), "correct": int(correct), "example_count": examples,
                "exact": int(exact), "transition_count": int(totals[TRANSITIONS]), "classes": classes}

    def observe(self, model, batch):
        """What a study measures at an observation beyond the metrics of the splits."""
        return {} if self.measures is None else self.measures.observe(model, batch)

    def inspect(self, model, batch):
        """What a study measures of the last and the best model."""
        return {} if self.measures is None else {"memorization": self.measures.inspect(model, batch)}

    def analyze(self, history):
        """The splits' fingerprints, the first observation whose selection `metric` reaches `target`, and the
        report of a study."""
        spec = self.spec
        hit = None if spec.target is None else next(
            (row for row in history if row[spec.selection][spec.metric] >= spec.target), None)
        found = {"split_fingerprints": self.fingerprints, "time_to_target": None if hit is None else {
            **{key: hit[key] for key in ("step", "training_seconds", "wall_seconds", "examples_seen")},
            "value": hit[spec.selection][spec.metric]}}
        if self.measures is not None:
            found["memorization"] = self.measures.report(history)
        return found
