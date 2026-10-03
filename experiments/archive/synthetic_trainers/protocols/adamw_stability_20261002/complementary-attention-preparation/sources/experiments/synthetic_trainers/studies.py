"""Training-time diagnostics for memorization, interpolation, and transfer."""

from .compression import code_lengths, membership_auc, prefix_extraction, uniform_compression
from .corpus import context_report, corpus_report, problem_key, study_pool
from .curves import curve_witness, delayed_generalization, fit_value
from .data import build_split
from .label_noise import noise_fit_metrics, noisy_split
from .metrics import evaluate
from .oracles import validate_example


def validate_study(spec, config):
    if spec.task == "random_lm":
        if config.label_noise:
            raise ValueError("random_lm already has IID random targets; do not add label noise")
        if config.split_policy != "independent":
            raise ValueError("random_lm requires IID sampling with replacement")


def prepare_study(spec, config, prepared=None):
    validate_study(spec, config)
    pool = prepared if prepared is not None else study_pool(spec, config)
    sizes = {"train": config.train_examples, "validation": config.validation_examples, "test": config.test_examples}
    if set(pool) != set(sizes):
        raise ValueError("A study pool must contain train, validation, and test")
    for name, split in pool.items():
        if split.spec != spec or split.seed != config.data_seed or split.name != name or len(split.examples) != sizes[name]:
            raise ValueError("Prepared study pool does not match the run configuration")
        for example in split.examples:
            validate_example(example, spec)
    overlap = corpus_report(pool)
    if config.split_policy == "disjoint" and (
            any(item["unique_inputs"] for item in overlap["overlaps"].values())
            or any(item["duplicate_rows"] for item in overlap["splits"].values())):
        raise ValueError("A disjoint study pool cannot contain duplicate or shared inputs")
    train, noise = noisy_split(pool["train"], config.label_noise, config.noise_seed)
    return pool, train, noise


class StudySession:
    def __init__(self, spec, config, pool, noise, train, eval_specs, model):
        self.spec, self.config, self.pool, self.noise, self.train = spec, config, pool, noise, train
        self.history = []
        self.parameters = sum(parameter.numel() for parameter in model.parameters())
        self.overlap = corpus_report(pool)
        self.contexts = context_report(train.examples)
        self.probes = {}
        for name, probe_spec in eval_specs.items():
            probe = build_split(probe_spec, "validation", config.data_seed, config.validation_examples)
            self.probes[name] = probe
        seen = {problem_key(row) for row in pool["train"].examples}
        self.novel_validation = tuple(row for row in pool["validation"].examples if problem_key(row) not in seen)
        self.novel_probes = {name: tuple(row for row in split.examples if problem_key(row) not in seen)
                             for name, split in self.probes.items()}

    def save(self, directory):
        self.pool["train"].save(directory / "data" / "train_clean")
        for name, split in self.probes.items():
            split.save(directory / "data" / "validation_probes" / name)

    def observe(self, model, record, device):
        config = self.config
        observed = evaluate(model, self.train.examples, config.batch_size, device, self.spec, free_generation=False)
        clean = (observed if not config.label_noise else evaluate(
            model, self.pool["train"].examples, config.batch_size, device, self.spec, free_generation=False))
        record.update({"train": observed, "train_clean": clean,
                       "train_evaluation": "teacher_forced", "epochs_seen": record["examples_seen"] / len(self.train.examples),
                       "validation_ood": {name: evaluate(model, split.examples, config.batch_size, device, split.spec)
                                          for name, split in self.probes.items()},
                       "fit_value": fit_value(observed, config.fit_metric)})
        def novel_score(rows, full_rows, score, spec):
            if not rows:
                return None
            return score if len(rows) == len(full_rows) else evaluate(model, rows, config.batch_size, device, spec)
        record["validation_novel"] = novel_score(self.novel_validation, self.pool["validation"].examples, record["validation"], self.spec)
        record["validation_ood_novel"] = {name: novel_score(rows, self.probes[name].examples,
                                                           record["validation_ood"][name], self.probes[name].spec)
                                           for name, rows in self.novel_probes.items()}
        if self.spec.task == "random_lm":
            scores = code_lengths(model, self.train.examples, config.batch_size, device)
            record["compression"] = uniform_compression(self.train.examples, scores, self.spec.symbols, self.parameters)
        if config.label_noise:
            record["noise_fit"] = noise_fit_metrics(model, self.train.examples, self.pool["train"].examples, config.batch_size, device)
        self.history.append(record)

    def checkpoint_diagnostics(self, model, device):
        config = self.config
        train_scores = code_lengths(model, self.pool["train"].examples, config.batch_size, device)
        validation_scores = code_lengths(model, self.pool["validation"].examples, config.batch_size, device)
        # Exclude identical complete inputs from the nonmember population.
        members = {problem_key(row) for row in self.pool["train"].examples}
        nonmembers = [score for row, score in zip(self.pool["validation"].examples, validation_scores)
                      if problem_key(row) not in members]
        report = {"loss_membership_auc": membership_auc(train_scores, nonmembers) if nonmembers else None,
                  "nonmember_examples": len(nonmembers), "member_examples": len(train_scores),
                  "membership_scope": "clean_answer_mean_loss; excludes_exact_input_overlap; validation_not_test"}
        if self.spec.task == "random_lm":
            report["train_compression"] = uniform_compression(self.pool["train"].examples, train_scores, self.spec.symbols, self.parameters)
            report["validation_compression"] = uniform_compression(self.pool["validation"].examples, validation_scores, self.spec.symbols, self.parameters)
            report["prefix_extraction"] = prefix_extraction(model, self.pool["train"].examples, config.batch_size, device)
            report["validation_prefix_extraction"] = prefix_extraction(model, self.novel_validation, config.batch_size, device) if self.novel_validation else None
        else:
            report["clean_answer_entropy_given_input_bits"] = 0
            report["compression_scope"] = "deterministic_oracle_answers; task_accuracy_is_not_a_capacity_in_bits"
        return report

    def report(self):
        config = self.config
        transition = delayed_generalization(self.history, config, random_control=self.spec.task == "random_lm")
        overlap = self.overlap["overlaps"]["train/validation"]["unique_inputs"]
        transition["train_validation_overlap_inputs"] = overlap
        transition["novel_validation_examples"] = len(self.novel_validation)
        transition["novel_probe_examples"] = {name: len(rows) for name, rows in self.novel_probes.items()}
        probe_overlap = corpus_report({"train": self.pool["train"], **self.probes})
        transition["probe_input_overlap"] = {name: probe_overlap["overlaps"][f"train/{name}"]["unique_inputs"] for name in self.probes}
        curves = {"validation_example_loss": [(row["step"], row["validation"]["example_loss"]) for row in self.history],
                  "validation_error": [(row["step"], 1 - row["validation"][config.target_metric]) for row in self.history]}
        if self.novel_validation:
            curves["validation_novel_error"] = [(row["step"], 1 - row["validation_novel"][config.target_metric]) for row in self.history]
        for name in self.probes:
            curves[f"{name}/example_loss"] = [(row["step"], row["validation_ood"][name]["example_loss"]) for row in self.history]
            curves[f"{name}/error"] = [(row["step"], 1 - row["validation_ood"][name][config.target_metric]) for row in self.history]
            if self.novel_probes[name]:
                curves[f"{name}/novel_error"] = [(row["step"], 1 - row["validation_ood_novel"][name][config.target_metric]) for row in self.history]
        witnesses = {name: curve_witness(points, config.curve_tolerance) for name, points in curves.items()}
        alignments = {}
        for name, witness in witnesses.items():
            if witness:
                peak = witness["points"][2][0]
                transition_step = transition["generalization_step"]
                alignments[name] = {"peak_step": peak, "generalization_step": transition_step,
                                    "transition_after_peak": transition_step is not None and transition_step > peak,
                                    "scope": "temporal_alignment_only; not_a_causal_test"}
        return {"profile": config.study, "noise": self.noise, "corpus": self.overlap,
                "causal_contexts": self.contexts, "generalization_transition": transition,
                "epoch_double_descent": witnesses, "peak_transition_alignment": alignments,
                "interpolation": {"metric": config.fit_metric, "epsilon": config.fit_epsilon,
                                  "first_observed_step": transition["observed_train_fit_step"],
                                  "final_value": self.history[-1]["fit_value"],
                                  "final_fitted": self.history[-1]["fit_value"] < config.fit_epsilon,
                                  "scope": "finite_pool_observed_fit; not_parameter_count_or_population_EMC"},
                "data_scope": "IID_uniform_payloads" if self.spec.task == "random_lm" else "synthetic_algorithmic_pool_with_paired_generators"}
