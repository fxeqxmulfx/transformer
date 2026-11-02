"""Unchanged synthetic data, with supervised logits gathered from the actual GPT.

Source: SyntheticTask and task160. GPT returns every position; gathering its
logits here preserves the original masks and cross-entropy objective. This
computes more readout positions than GPTMini's optimized positional interface.
Generation, sequence scoring, data streams and oracle validation are inherited
unchanged. The benchmark reserves all final-test splits until cohort closure.
"""

from .synthetic.training import SyntheticTask
from .synthetic.vocabulary import IGNORE


class BasisControlTask(SyntheticTask):
    """Original GPTMini readout with task160's reserved-test endpoint summary.

    Sources: SyntheticTask and ResidualBasisTask at23154349. No change to
    the objective or original model interface; only endpoint reporting differs.
    """

    def analyze(self, history):
        name, metric = self.spec.validation_name, self.spec.metric
        first = next((row for row in history if row[name][metric] >= self.spec.target), None)
        return {"split_fingerprints": self.fingerprints,
                "time_to_target": None if first is None else {
                    **{key: first[key] for key in ("step", "training_seconds", "wall_seconds", "examples_seen")},
                    "value": first[name][metric]}}


class ResidualBasisTask(BasisControlTask):
    def forward(self, model, batch, supervised=False):
        tokens, targets = batch[:, 0], batch[:, 1]
        logits = model(tokens)
        if not supervised:
            return logits, targets
        positions = (targets != IGNORE).int().argsort(dim=1, descending=True, stable=True)
        positions = positions[:, :self.splits["train"].readout]
        return logits.gather(1, positions[..., None].expand(-1, -1, logits.shape[-1])), targets.gather(1, positions)
