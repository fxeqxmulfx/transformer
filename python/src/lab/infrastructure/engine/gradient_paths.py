"""Prepare known-solution runs without rounding their saved Schedule-Free frame.

Source: task145's complete-state preparation guard. No conditioning transform
or custom optimizer is used; learning is ordinary lab Schedule-Free AdamW.
"""

import copy

from .compiled import CompiledStepper
from .eager import EagerStepper


class PreservePreparation:
    def prepare(self, sizes):
        state = copy.deepcopy((self.model.state_dict(), self.optimizer.state_dict()))
        training = self.model.training
        try:
            return super().prepare(sizes)
        finally:
            self.model.load_state_dict(state[0])
            self.optimizer.load_state_dict(state[1])
            self.model.train(training)


class GradientCompiledStepper(PreservePreparation, CompiledStepper):
    """Fullgraph max-autotune with complete-state preparation semantics."""


class GradientEagerStepper(PreservePreparation, EagerStepper):
    """Independent eager reference using the same complete-state guard."""
