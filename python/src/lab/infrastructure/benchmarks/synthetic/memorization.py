"""What a memorization study measures beyond the metrics of its splits.

A port of `StudySession` of `experiments/synthetic_trainers/studies.py`. An
observation of the random control adds the compression of its training
answers, and one under label noise how the model fits the corrupted labels
(`observe`). A trained model, the last and the best, is measured for the
membership of its training rows, and the random control's for compression
and extraction too (`inspect`, `checkpoint_diagnostics`). The report sets
the facts of the splits beside what their history shows (`report`).
"""

from ....domain.generative import RandomLM
from ....domain.memorization import report
from .compression import code_lengths, membership_auc, prefix_extraction, uniform_compression
from .corpus import context_report, corpus_report
from .noise import noise_fit
from .rows import Rows


def parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


class Measures:
    """A study's novel rows, and its measurements of models and of a history.

    It adds to `rows`, the task's rows by split, the rows of validation and
    of each held-out split whose input no training row has: the split's own
    rows when every input is new, and None when none is.
    """

    def __init__(self, benchmark, splits, noise, rows, device):
        self.benchmark, self.noise, self.rows, self.device = benchmark, noise, rows, device
        self.control = isinstance(benchmark.task, RandomLM)
        clean = splits["train_clean"]
        self.key = clean.generator.key
        self.members = set(map(self.key, clean.examples))
        self.novel = {}
        for name in ("validation", *(f"validation/{probe}" for probe in benchmark.probes)):
            examples = splits[name].examples
            novel = tuple(example for example in examples if not self.member(example))
            self.novel[name] = novel
            rows[f"{name}/novel"] = (None if not novel else rows[name] if len(novel) == len(examples)
                                     else Rows(novel, device))
        self.corpus = corpus_report({"train": clean, "validation": splits["validation"], "test": splits["test"]})
        self.contexts = context_report(splits["train"].examples)
        probes = corpus_report({"train": clean, **{probe: splits[f"validation/{probe}"] for probe in benchmark.probes}})
        self.probe_overlap = {probe: probes["overlaps"][f"train/{probe}"]["unique_inputs"]
                              for probe in benchmark.probes}

    def member(self, example):
        """Whether a training row has the input of `example`; the splits of one task key their inputs alike."""
        return self.key(example) in self.members

    def compression(self, model, examples, scores):
        if scores is None:
            return None
        return uniform_compression(examples, scores, self.benchmark.task.symbols, parameters(model))

    def observe(self, model, batch):
        """The compression of the random control's training answers, and the fit of corrupted labels."""
        measured = {}
        if self.control:
            rows = self.rows["train"]
            measured["compression"] = self.compression(model, rows.examples, code_lengths(model, rows, batch))
        if self.benchmark.study.noise:
            measured["noise_fit"] = noise_fit(model, self.rows["train"], self.rows["train_clean"], batch)
        return measured

    def inspect(self, model, batch):
        """How much better a trained model codes its training answers than novel validation answers.

        The membership AUC compares the bits per target of the oracle's
        training answers with those of validation rows with novel inputs;
        the random control's model is also measured for compression on both,
        and for extraction of training and of novel validation answers.
        """
        clean, validation = self.rows["train_clean"], self.rows["validation"]
        members, scores = code_lengths(model, clean, batch), code_lengths(model, validation, batch)
        novel = [not self.member(example) for example in validation.examples]
        nonmembers = None if scores is None else [score for score, new in zip(scores, novel) if new]
        measured = {"loss_membership_auc": membership_auc(members, nonmembers) if members and nonmembers else None,
                    "nonmember_examples": sum(novel), "member_examples": len(clean),
                    "membership_scope": "clean_answer_mean_loss; excludes_exact_input_overlap; validation_not_test"}
        if not self.control:
            return {**measured, "clean_answer_entropy_given_input_bits": 0,
                    "compression_scope": "deterministic_oracle_answers; task_accuracy_is_not_a_capacity_in_bits"}
        held_out = self.novel["validation"]
        return {**measured, "train_compression": self.compression(model, clean.examples, members),
                "validation_compression": self.compression(model, validation.examples, scores),
                "prefix_extraction": prefix_extraction(model, clean.examples, batch, self.device),
                "validation_prefix_extraction": prefix_extraction(model, held_out, batch, self.device) if held_out
                else None}

    def report(self, history):
        """The noise, the inputs the splits hold and share, and what the history shows (`StudySession.report`)."""
        shown = report(history, self.benchmark)
        shown["generalization_transition"].update({
            "train_validation_overlap_inputs": self.corpus["overlaps"]["train/validation"]["unique_inputs"],
            "novel_validation_examples": len(self.novel["validation"]),
            "novel_probe_examples": {probe: len(self.novel[f"validation/{probe}"]) for probe in self.benchmark.probes},
            "probe_input_overlap": self.probe_overlap})
        return {"noise": self.noise, "corpus": self.corpus, "causal_contexts": self.contexts, **shown}
