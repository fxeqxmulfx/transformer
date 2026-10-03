"""Masked loss and counts weighted by actual answers, including partial batches."""

import torch
from torch.nn import functional as F

from .vocabulary import IGNORE, token_name


def masked_loss(logits, targets):
    if logits.shape[:-1] != targets.shape:
        raise ValueError("Logit and target shapes disagree")
    mask = targets != IGNORE
    if not mask.any():
        raise ValueError("No supervised targets")
    return F.cross_entropy(logits[mask], targets[mask])


class Metrics:
    def __init__(self, spec=None):
        self.spec = spec
        self.targets = self.correct = self.examples = self.exact = 0
        self.transitions = self.transition_correct = 0
        self.loss_sum = 0.0
        self.example_loss_sum = 0.0
        self.classes = {}

    def add(self, logits, batch):
        valid = batch.targets != IGNORE
        if not valid.any(dim=1).all():
            raise ValueError("Every example needs a supervised answer")
        if logits.shape[:-1] != batch.targets.shape:
            raise ValueError("Logit and target shapes disagree")
        token_losses = F.cross_entropy(logits[valid].float(), batch.targets[valid], reduction="none")
        if not torch.isfinite(token_losses).all():
            raise RuntimeError("Nonfinite evaluation loss")
        loss = token_losses.mean()
        correct = logits.argmax(-1) == batch.targets
        count = int(valid.sum())
        self.targets += count
        self.correct += int((correct & valid).sum())
        self.loss_sum += float(loss) * count
        rows = torch.arange(len(batch.tokens), device=valid.device)[:, None].expand_as(valid)[valid]
        example_losses = torch.zeros(len(batch.tokens), device=valid.device, dtype=token_losses.dtype).scatter_add_(0, rows, token_losses)
        self.example_loss_sum += float((example_losses / valid.sum(1)).sum())
        self.examples += len(batch.tokens)
        self.exact += int((correct | ~valid).all(dim=1).sum())
        self.transitions += int(batch.changes.sum())
        self.transition_correct += int((correct & batch.changes).sum())
        for label in batch.targets[valid].unique().tolist():
            selected = batch.targets == label
            counts = self.classes.setdefault(label, [0, 0])
            counts[0] += int(selected.sum())
            counts[1] += int((selected & correct).sum())

    def report(self):
        if not self.targets:
            raise ValueError("Cannot report empty metrics")
        classes = {str(label): {"name": token_name(label, self.spec), "count": count,
                               "correct": correct, "accuracy": correct / count}
                   for label, (count, correct) in sorted(self.classes.items())}
        return {
            "loss": self.loss_sum / self.targets,
            "example_loss": self.example_loss_sum / self.examples,
            "token_accuracy": self.correct / self.targets,
            "sequence_accuracy": self.exact / self.examples,
            "balanced_accuracy": sum(item["accuracy"] for item in classes.values()) / len(classes),
            "transition_accuracy": self.transition_correct / self.transitions if self.transitions else None,
            "target_count": self.targets, "correct": self.correct,
            "example_count": self.examples, "exact": self.exact,
            "transition_count": self.transitions, "classes": classes,
        }


@torch.no_grad()
def evaluate(model, examples, batch_size, device="cpu", spec=None, *, free_generation=True):
    from .records import collate

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not examples or len({bool(example.prompt) for example in examples}) != 1:
        raise ValueError("Evaluation needs one nonempty supervision format")
    previous = model.training
    model.eval()
    metrics = Metrics(spec)
    generative = bool(examples[0].prompt) and free_generation
    if generative:
        from .generation import rollout
        from .generation_metrics import GenerationMetrics

        generated = GenerationMetrics(spec)
    try:
        for start in range(0, len(examples), batch_size):
            batch = collate(examples[start:start + batch_size], device)
            metrics.add(model(batch.tokens), batch)
            if generative:
                rows = examples[start:start + batch_size]
                result = rollout(model, [row.prompt for row in rows],
                                 [row.generation_limit for row in rows], device)
                generated.add(rows, result)
        teacher = metrics.report()
        if not generative:
            return teacher
        return {**generated.report(), "loss": teacher["loss"], "example_loss": teacher["example_loss"],
                "teacher_forced": teacher}
    finally:
        model.train(previous)


def validation_rank(metrics, primary="sequence_accuracy"):
    if primary == "final_answer_accuracy":
        return metrics[primary], metrics["sequence_accuracy"], metrics["balanced_accuracy"], -metrics["loss"]
    return metrics["sequence_accuracy"], metrics["balanced_accuracy"], -metrics["loss"]
