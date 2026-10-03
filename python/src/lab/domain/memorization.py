"""Memorization studies: finite pools, label noise, and what the history of a study shows.

A port of the study profiles of the historical synthetic suite
(`experiments/synthetic_trainers/studies.py`, `corpus.py`, `curves.py`).
`report` reads a study's observations: when training fits, when novel and
held-out inputs are generalized to, and where a curve descends twice. Each
is a finite witness of the observations, not a causal claim.
"""

from dataclasses import dataclass
import math

from .analysis import curve_witness
from .generative import RandomLM
from .spec import Spec, require

FITS = ("example_error", "token_error", "loss")


@dataclass(frozen=True)
class Memorization(Spec):
    """The finite pools of a memorization study, the label noise of its training pool, and its criteria.

    Source: the study profiles of the historical synthetic suite
    (`experiments/synthetic_trainers/studies.py`, `corpus.py`, `sweeps.py`),
    `memorization` and `double_descent`, which ran this same study under two
    names. With `disjoint` no input recurs within or across the train,
    validation and test pools; otherwise each is sampled on its own, with
    replacement. `noise` replaces each training label that has another legal
    value, with that probability, by one of them drawn uniformly, once per
    input and position from `noise_seed` (arXiv:1912.02292v1, Section 4);
    BOS, EOS, separators and index hints keep their labels. A `pool` larger
    than the training split is sampled in full and training takes its first
    rows, so runs on nested training pools share their validation and test
    pools.

    Training fits when the `fit` error of teacher forcing on its observed
    labels, the example error, the token error or the example loss, is below
    `epsilon`. Novel inputs are generalized to at the first of `patience`
    consecutive observations at which validation, on the inputs no training
    row has, reaches the benchmark's target, and transferred to when every
    held-out distribution does too. A double descent of a curve rises and
    falls by more than `tolerance`.
    """
    disjoint: bool = False
    noise: float = 0.0
    noise_seed: int = 0
    pool: int | None = None
    fit: str = "example_error"
    epsilon: float = 0.01
    patience: int = 2
    tolerance: float = 0.001

    def check(self):
        require(math.isfinite(self.noise) and 0 <= self.noise <= 1, "Label noise is a probability")
        require(self.noise_seed >= 0, "The noise seed is nonnegative")
        require(self.pool is None or self.pool >= 1, "A pool holds at least one input")
        require(self.fit in FITS, f"The fit is one of {', '.join(FITS)}")
        require(math.isfinite(self.epsilon) and 0 < self.epsilon < 1, "The fit threshold lies strictly in (0, 1)")
        require(self.patience >= 1, "Generalization is confirmed by at least one observation")
        require(math.isfinite(self.tolerance) and self.tolerance >= 0, "The curve tolerance is finite and nonnegative")


def fit_value(metrics, fit):
    """The `fit` error of teacher-forced metrics (`curves.fit_value`)."""
    if fit == "loss":
        return metrics["example_loss"]
    return 1 - metrics["sequence_accuracy" if fit == "example_error" else "token_accuracy"]


def at(row, key):
    return None if row is None else row[key]


def delayed_generalization(history, benchmark):
    """When a study's training first fits, and when novel inputs are first generalized and transferred to.

    A port of `curves.delayed_generalization`. The lag is that of
    arXiv:2211.11052v1, Section 4, from the first fit of the oracle's labels;
    transfer adds every held-out distribution (`Synthetic.probes`) to the
    criterion. The random control has no answer to generalize to.
    """
    study, target, metric = benchmark.study, benchmark.target, benchmark.metric
    control = isinstance(benchmark.task, RandomLM)

    def fitted(split):
        return next((row for row in history if fit_value(row[split], study.fit) < study.epsilon), None)

    def onset(transfer):
        """The first of `patience` consecutive qualifying observations, and the last of them."""
        start, streak = None, 0
        for row in history:
            validation = row["validation/novel"]
            probes = [row[f"validation/{name}/novel"] for name in benchmark.probes]
            qualifies = not control and target is not None and validation is not None and validation[metric] >= target
            if transfer:
                qualifies = qualifies and bool(probes) and all(
                    probe is not None and probe[metric] >= target for probe in probes)
            if not qualifies:
                start, streak = None, 0
                continue
            streak += 1
            if streak == 1:
                start = row
            if streak >= study.patience:
                return start, row
        return None, None

    observed, clean = fitted("train"), fitted("train_clean")

    def since_fit(row, key):
        return None if row is None or clean is None else row[key] - clean[key]

    report = {"observed_train_fit_step": at(observed, "step"), "clean_train_fit_step": at(clean, "step")}
    for prefix, candidate, transfer in (("", "delayed_transfer_candidate", True),
                                        ("id_", "delayed_id_generalization_candidate", False)):
        start, confirmation = onset(transfer)
        lag = since_fit(start, "step")
        report.update({f"{prefix}generalization_step": at(start, "step"),
                       f"{prefix}confirmed_at_step": at(confirmation, "step"), f"{prefix}lag_steps": lag,
                       f"{prefix}lag_epochs": since_fit(start, "epochs_seen"),
                       f"{prefix}lag_training_seconds": since_fit(start, "training_seconds"),
                       candidate: lag is not None and lag > 0})
    return {**report, "criterion": {
        "fit_metric": study.fit, "fit_epsilon": study.epsilon, "validation_target": target,
        "target_metric": metric, "consecutive_observations": study.patience,
        "requires_all_ood_probes": True, "requires_novel_inputs": True},
        "scope": "scheduled_validation_observations; transfer_evidence_not_proof_of_understanding"}


def report(history, benchmark):
    """What a study's observations show, as `StudySession.report` read them from its history.

    Each validation and held-out curve of example loss and of error (one
    minus the target metric), and of error on novel inputs where there are
    any, is searched for a double descent (`curve_witness`), and the peak of
    each witness is set beside the transfer step: an order in time only.
    """
    study, metric = benchmark.study, benchmark.metric
    transition = delayed_generalization(history, benchmark)

    def curves(split, name):
        found = {f"{name}example_loss": [(row["step"], row[split]["example_loss"]) for row in history],
                 f"{name}error": [(row["step"], 1 - row[split][metric]) for row in history]}
        if history[0][f"{split}/novel"] is not None:
            found[f"{name}novel_error"] = [(row["step"], 1 - row[f"{split}/novel"][metric]) for row in history]
        return found

    found = curves("validation", "validation_")
    for name in benchmark.probes:
        found.update(curves(f"validation/{name}", f"{name}/"))
    witnesses = {name: curve_witness(points, study.tolerance) for name, points in found.items()}
    step = transition["generalization_step"]
    alignments = {name: {"peak_step": witness["points"][2][0], "generalization_step": step,
                         "transition_after_peak": step is not None and step > witness["points"][2][0],
                         "scope": "temporal_alignment_only; not_a_causal_test"}
                  for name, witness in witnesses.items() if witness}
    final = fit_value(history[-1]["train"], study.fit)
    return {"generalization_transition": transition, "epoch_double_descent": witnesses,
            "peak_transition_alignment": alignments,
            "interpolation": {"metric": study.fit, "epsilon": study.epsilon,
                              "first_observed_step": transition["observed_train_fit_step"],
                              "final_value": final, "final_fitted": final < study.epsilon,
                              "scope": "finite_pool_observed_fit; not_parameter_count_or_population_EMC"}}
