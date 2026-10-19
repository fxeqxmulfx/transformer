"""The source-matched pairs with scalar/vector squared loss or categorical CE.

Source: landscape_grokking_plan.md, M9.3; LabDivisionTask at c59d4426 for
every arithmetic row, split and sampler. Half squared error is averaged per
equation and sums vector coordinates. Numeric error uses decoded residue
units; scalar heads also retain their unrounded numeric error. Observers
measure conditioning on a fixed training prefix and do not change updates.
"""

import torch
from torch.nn import functional as F

from ...domain.paired import CrossEntropyAnswers
from .rotary import LabDivisionTask


class PairTask(LabDivisionTask):
    def __init__(self, spec, data_seed, device, vector):
        super().__init__(spec.data, data_seed, device)
        self.objective, self.vector = spec.loss, vector

    def summary(self):
        return {**super().summary(), "answer_objective": type(self.objective).__name__,
                "readout": "vector" if self.vector else "scalar",
                "numeric_error_units": "residue_value",
                "target_coding": ("class_index" if isinstance(self.objective, CrossEntropyAnswers)
                                  else "one_hot" if self.vector else "2q/(p-1)-1")}

    @staticmethod
    def forward(model, batch, supervised=False):
        return model(batch[:, :2]), batch[:, 2]

    def row_losses(self, output, labels):
        if isinstance(self.objective, CrossEntropyAnswers):
            return F.cross_entropy(output, labels, reduction="none")
        target = (F.one_hot(labels, self.vocab).to(output.dtype) if self.vector
                  else labels.to(output.dtype) * (2 / (self.vocab - 1)) - 1)
        squared = (output - target).square()
        return 0.5 * (squared.sum(dim=-1) if self.vector else squared)

    def loss(self, output, labels):
        return self.row_losses(output, labels).mean()

    def position_losses(self, output, labels):
        return self.loss(output, labels).reshape(1)

    def accumulator(self):
        return torch.zeros(4, dtype=torch.float64, device=self.device)

    def accumulate(self, model, chunk, sums, static):
        output, labels = self.forward(model, chunk)
        if self.vector:
            answer = output.argmax(dim=-1)
            continuous = answer.double()
        else:
            continuous = (output + 1) * ((self.vocab - 1) / 2)
            answer = continuous.round()
        decoded_error = answer.double() - labels.double()
        continuous_error = continuous.double() - labels.double()
        sums.add_(torch.stack((self.row_losses(output, labels).sum().double(),
                               (answer == labels).sum().double(), decoded_error.square().sum(),
                               continuous_error.square().sum())))

    def metrics(self, sums, rows):
        loss, correct, squared, continuous = sums
        count = len(rows)
        metrics = {"loss": loss / count, "accuracy": correct / count,
                   "answer_loss": loss / count, "answer_accuracy": correct / count,
                   "numeric_mse": squared / count, "examples": count}
        if not self.vector:
            metrics["continuous_numeric_mse"] = continuous / count
        return metrics

    @torch.no_grad()
    def observe(self, model, batch):
        probe = self.splits["train"][:batch, :2]
        first = model.hidden[0](model.encode(probe))
        return {"parameter_norm": float(torch.sqrt(sum(p.detach().square().sum() for p in model.parameters()))),
                "first_layer_probe_examples": len(probe),
                "first_preactivation_rms": float(first.square().mean().sqrt()),
                "first_activation_rms": float(model.activation(first).square().mean().sqrt())}
