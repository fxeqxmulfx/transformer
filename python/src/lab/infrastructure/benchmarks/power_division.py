"""Powered division, with independent multiplication certificates for every label.

Source: Transformer.Rotate90.existsUnique_div_pow at 1c9b23b2; the parent
LabDivisionTask supplies the established split, sampler, loss and metrics.
The exponent transforms only labels, so operand splits match across powers.
"""

import torch

from .rotary import LabDivisionTask, fingerprint


class PowerDivisionTask(LabDivisionTask):
    def __init__(self, spec, data_seed, device):
        super().__init__(spec, data_seed, device)
        powers = torch.tensor([pow(q, spec.exponent, spec.prime) for q in range(spec.prime)],
                              device=device, dtype=torch.long)
        for rows in self.splits.values():
            rows[:, 2] = powers[rows[:, 2]]
            if any(pow(b, spec.exponent, spec.prime) * c % spec.prime != pow(a, spec.exponent, spec.prime)
                   for a, b, c in rows.tolist()):
                raise ValueError("The powered multiplication oracle rejected a label")
        self.metadata = self.summary()

    def summary(self):
        summary = super().summary()
        summary["exponent"] = self.spec.exponent
        summary["label_oracle"] = "pow(b,k,p)*c % p == pow(a,k,p)"
        summary["operand_train_fingerprint"] = fingerprint(self.splits["train"][:, :2])
        summary["operand_heldout_fingerprint"] = fingerprint(self.splits["heldout"][:, :2])
        summary["label_classes"] = len({pow(q, self.spec.exponent, self.vocab) for q in range(self.vocab)})
        for split, rows in self.splits.items():
            counts = torch.bincount(rows[:, 2], minlength=self.vocab)
            summary[f"{split}_majority_accuracy"] = float(counts.max()) / len(rows) if len(rows) else None
        summary["scope"] = ("full_table_fitting_only" if self.spec.train_fraction == 1
                            else "unseen_pairs_at_each_fixed_prime_and_power")
        return summary

    def observe(self, model, batch):
        return {"parameter_norm": float(torch.sqrt(sum(p.detach().square().sum() for p in model.parameters()))),
                "parameters": sum(p.numel() for p in model.parameters()), "task": self.metadata}

    @staticmethod
    def analyze(history):
        result = {}
        for split in ("train", "heldout"):
            points = [point for point in history if split in point and point[split] is not None]
            result[f"first_{split}99_update"] = next(
                (point["step"] for point in points if point[split]["accuracy"] >= 0.99), None)
            result[f"best_{split}_accuracy"] = max((point[split]["accuracy"] for point in points), default=None)
        return result
