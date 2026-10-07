"""Complete raw-data Basis supervision, absent from model inference.

Source: Structured.DepthReference, ParityReference, RecallDataTargets and
basisStackBatchNLL_convex at b0a43a8. Deviations: tensorized finite scans
and latest-record selection; auxiliary labels are precomputed once from
the unchanged training rows. Ordinary output metrics remain inherited.
"""

import torch

from ....domain.generative import Parity
from ....domain.tasks import AlternatingBlocks
from ...arithmetic import Arithmetic
from .training import SyntheticTask
from .vocabulary import A, B, BOS, EVEN, IGNORE, NEUTRAL, ODD, ONE, SEP, ZERO


def depth_step(token, state):
    """Independent raw data recurrence; never called by attention."""
    if token == BOS:
        return 0
    if token == NEUTRAL:
        return state
    if token == A:
        return {0: 1, 2: 3, 4: 5}.get(state, state)
    if token == B:
        return {0: 5, 1: 2, 3: 4}.get(state, state)
    return 5


def parity_step(token, state):
    """Independent bit/SEP/label phases, including the second EOS call."""
    if token == BOS:
        return 0
    if token == ZERO:
        return state if state in (0, 1) else 5
    if token == ONE:
        return 1 - state if state in (0, 1) else 5
    if token == SEP:
        return state + 2 if state in (0, 1) else 5
    if token == EVEN and state == 2 or token == ODD and state == 3:
        return 4
    return 5


def reference_table(task, vocab, device):
    """Tensorized raw reference table; every arithmetic operation is traceable."""
    token = torch.arange(vocab, device=device)[:, None]
    state = torch.arange(6, device=device)[None, :]
    if isinstance(task, AlternatingBlocks):
        on_a = torch.where(state == 0, 1, torch.where(state == 2, 3, torch.where(state == 4, 5, state)))
        on_b = torch.where(state == 0, 5, torch.where(state == 1, 2, torch.where(state == 3, 4, state)))
        table = torch.where(token == A, on_a, torch.where(token == B, on_b,
                            torch.where(token == NEUTRAL, state, 5)))
    else:
        valid = state <= 1
        on_zero, on_one = torch.where(valid, state, 5), torch.where(valid, 1 - state, 5)
        on_sep = torch.where(valid, state + 2, 5)
        answered = ((token == EVEN) & (state == 2)) | ((token == ODD) & (state == 3))
        table = torch.where(token == ZERO, on_zero, torch.where(token == ONE, on_one,
                            torch.where(token == SEP, on_sep, torch.where(answered, 4, 5))))
    return torch.where(token == BOS, 0, table)


def complete_labels(tokens, targets, task, vocab):
    """Actual state histories or latest physical routes from unchanged rows."""
    if isinstance(task, (AlternatingBlocks, Parity)):
        table = reference_table(task, vocab, tokens.device)
        state = torch.zeros(len(tokens), dtype=torch.long, device=tokens.device)
        previous, history = [], []
        for token in tokens.unbind(1):
            previous.append(state)
            state = table[token, state]
            history.append(state)
        return torch.stack((torch.stack(previous, 1), torch.stack(history, 1), targets.clamp_min(0)), 1)
    position = torch.arange(task.pairs, device=tokens.device) * 2 + 2
    keys = tokens[:, position - 1]
    matching = tokens[..., None] == keys[:, None, :]
    value = torch.where(matching, position, 0).amax(-1)
    key = (value - 1).clamp_min(0)
    return torch.stack((key, value, targets.clamp_min(0)), 1)


class TensorBasisTask(SyntheticTask):
    """Same Basis splits/evaluation; actual complete likelihood for training."""
    def __init__(self, spec, data_seed, device):
        super().__init__(spec, data_seed, device)
        self.branch = 0 if isinstance(spec.task, (AlternatingBlocks, Parity)) else 1
        rows = self.splits["train"]
        counter = Arithmetic()
        with counter:
            self.observed = complete_labels(rows.rows[:, 0], rows.rows[:, 1], spec.task, self.vocab)
        self.label_arithmetic = counter.report()

    def inputs(self, indices, static=False):
        batch = super().inputs(indices, static)
        observed = self.observed.index_select(0, indices.to(self.device))[:, :, :batch.shape[-1]]
        return torch.cat((batch, observed), 1)

    def forward(self, model, batch, supervised=False):
        losses = model.complete_losses(batch[:, 0], batch[:, 3:6], self.branch)
        targets = batch[:, 1]
        if supervised:
            positions = (targets != IGNORE).int().argsort(dim=1, descending=True, stable=True)
            positions = positions[:, :self.splits["train"].readout]
            return losses.gather(1, positions), targets.gather(1, positions)
        return losses, targets

    @staticmethod
    def loss(output, targets):
        valid = targets != IGNORE
        return output.masked_fill(~valid, 0).sum() / valid.sum()

    @staticmethod
    def position_losses(output, targets):
        return TensorBasisTask.loss(output, targets).reshape(1)
