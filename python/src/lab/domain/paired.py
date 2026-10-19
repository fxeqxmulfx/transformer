"""Matched input, readout and loss controls for the browser's division pairs.

Source: landscape_grokking_plan.md, M9.3; the scalar perceptron at 897e80be
and the source-matched LabDivision protocol at c59d4426. A vector readout
adds one free output per residue. SquaredAnswers supports either head;
CrossEntropyAnswers changes only the objective of the same vector model.
"""

from dataclasses import dataclass

from .benchmarks import Benchmark
from .model import Model
from .regression import MLP
from .rotary import LabDivision
from .spec import Spec, require, require_kind


@dataclass(frozen=True)
class AnswerLoss(Spec, kind=True):
    """The per-equation objective of a DivisionPairs experiment (M9.3)."""


@dataclass(frozen=True)
class SquaredAnswers(AnswerLoss):
    """Half the squared Euclidean error, averaged over equations.

    A scalar head targets 2q/(p-1)-1; a vector head targets the one-hot e_q.
    Vector coordinates are summed, so this does not divide by vocabulary.
    Source: M9.3's readout control; the factor 1/2 differs from the old
    whole-table ModularRegression and makes the residual its logit gradient.
    """


@dataclass(frozen=True)
class CrossEntropyAnswers(AnswerLoss):
    """Full p-way cross-entropy on raw vector logits, without an EOS term.

    Source: the recorded Rotate90 objective, applied to an untied MLP head.
    """


@dataclass(frozen=True)
class VectorReadout(Model):
    """Replace an MLP's final scalar projection by p independent raw outputs.

    Source: M9.3's readout control. Hidden layers, biases and the initialization
    specification are retained; the vocabulary fixes the output width. With
    TorchDefault the hidden parameter draws are identical across heads.
    SquaredAnswers or CrossEntropyAnswers supplies the objective separately.
    """
    network: MLP

    def check(self):
        require_kind(self.network, MLP, "network")

    @property
    def context(self):
        return self.network.context


@dataclass(frozen=True)
class DivisionPairs(Benchmark):
    """The exact LabDivision equations/split/sampler with a specified answer loss.

    Source: grokking_abmod f49969393a7482490d45483d8a4fd3046b1adede,
    browser arithmetic protocol, ported at c59d4426. Unlike the old scalar
    regression study, these controls retain both train and held-out splits.
    """
    data: LabDivision
    loss: AnswerLoss

    def check(self):
        require_kind(self.data, LabDivision, "data")
        require_kind(self.loss, AnswerLoss, "loss")
        require(isinstance(self.loss, (SquaredAnswers, CrossEntropyAnswers)), "Unknown answer objective")

    @property
    def context(self):
        return self.data.context

    @property
    def observed(self):
        return self.data.observed


def check_division_model(model, benchmark):
    """Check the explicit scalar/vector output contract before construction."""
    require(isinstance(model, (MLP, VectorReadout)), "DivisionPairs needs an MLP or its VectorReadout")
    require(model.context == benchmark.context, "The division MLP reads exactly two operands")
    require(not isinstance(benchmark.loss, CrossEntropyAnswers) or isinstance(model, VectorReadout),
            "CrossEntropyAnswers needs a VectorReadout")
