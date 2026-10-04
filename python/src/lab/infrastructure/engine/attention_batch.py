"""The fixed validation inputs and recall routes of EXPERIMENT_PLAN.md, step 2.

Synthetic examples keep their original causal context, including BOS and
teacher-forced answer tokens, but exclude padding. MQAR's answer column is
the latest value written for the queried key, as in
`benchmarks.synthetic.retrieval.MQAR.targets`; unrelated occurrences of
the same value are not answer columns. Modular division uses heldout and
its six-token teacher-forced context (`benchmarks.modular.ModularTask`).
"""

import hashlib

import torch

from ...domain.benchmarks import ModularDivision
from ...domain.synthetic import Synthetic
from ...domain.tasks import MQAR
from ..benchmarks.synthetic.vocabulary import IGNORE


def recall_queries(tokens, targets, pairs):
    """The last write's value column for every supervised MQAR query, in example/position order."""
    queries = []
    for example, (context, labels) in enumerate(zip(tokens, targets, strict=True)):
        latest = {context[position]: position + 1 for position in range(1, 1 + 2 * pairs, 2)}
        for position, target in enumerate(labels):
            if target == IGNORE:
                continue
            answer = latest[context[position]]
            if answer >= position or context[answer] != target:
                raise ValueError("An MQAR query must read its key's latest preceding value")
            queries.append({"example": example, "query_position": position,
                            "answer_position": answer, "target": target})
    return queries


class FixedBatch:
    """The first up to `examples` rows of one split, chosen without drawing random numbers."""

    def __init__(self, task, examples):
        self.queries = []
        spec = task.spec
        if isinstance(spec, Synthetic):
            self.split = spec.selection
            rows = task.splits[self.split]
            count = min(examples, len(rows))
            lengths = rows.lengths[:count]
            self.batch = rows.rows[:count, :, :max(lengths)]
            self.targets = self.batch[:, 1]
            self.supervised = self.targets != IGNORE
            if isinstance(spec.task, MQAR):
                self.queries = recall_queries(self.batch[:, 0].cpu().tolist(), self.targets.cpu().tolist(),
                                              spec.task.pairs)
        elif isinstance(spec, ModularDivision):
            self.split = "heldout"
            self.batch = task.splits[self.split][:examples]
            lengths = [self.batch.shape[1] - 1] * len(self.batch)
            self.targets = self.batch[:, 5:]
            self.supervised = torch.zeros((len(self.batch), lengths[0]), dtype=torch.bool, device=task.device)
            self.supervised[:, 4:] = True
        else:
            raise ValueError("Attention observations need a synthetic or modular-division task")
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=task.device)
        self.fingerprint = hashlib.sha256(self.batch.contiguous().cpu().numpy().tobytes()).hexdigest()

    def predictions(self, output):
        """Teacher-forced predictions at the supervised positions and at each MQAR query."""
        predicted = output.argmax(-1)
        valid = self.targets != IGNORE
        correct = (predicted == self.targets) & valid
        queries = []
        for query in self.queries:
            prediction = int(predicted[query["example"], query["query_position"]])
            queries.append({**query, "prediction": prediction, "correct": prediction == query["target"]})
        return {"supervised_tokens": int(valid.sum()),
                "teacher_forced_supervised_accuracy": float(correct.sum() / valid.sum()), "queries": queries}


def answer_routes(weights, queries):
    """Each query's answer-column weight per head, its maximum and membership in any support."""
    if not queries:
        return {"weights_per_head": [], "maximum_weight": [], "in_any_support": []}
    device = weights.device
    examples, positions, answers = (torch.tensor([query[key] for query in queries], device=device)
                                    for key in ("example", "query_position", "answer_position"))
    values = weights[examples, :, positions, answers]
    return {"weights_per_head": values.cpu().tolist(), "maximum_weight": values.max(-1).values.cpu().tolist(),
            "in_any_support": (values > 0).any(-1).cpu().tolist()}
