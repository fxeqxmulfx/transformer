"""Synthetic benchmarks sample the historical splits, row for row.

Golden records: `fixtures/legacy_splits.json`. Its `runs` are every distinct
split configuration of the two archived synthetic suites
(`experiments/archive/synthetic_trainers/baselines/amsgradw_softmax_20261002`
and `amsgradw_softmax_scaling_20261002`), with the fingerprints and the number
of corrupted labels those runs recorded; its `grid` was computed by the
historical generator, at the commit the fixture names, on controls the runs
leave unexercised. A fingerprint hashes every row of a split with the
historical task specification and seed, so an equal fingerprint is an equal
split.
"""

from dataclasses import fields
import json
from pathlib import Path
import random
import unittest

from lab.domain.generative import (Addition, BooleanAnd, Copy, Count, DoubleHistogram, Histogram, Mode,
                                   MostFrequent, Parity, RandomLM, Reverse, Sort)
from lab.domain.memorization import Memorization
from lab.domain.spec import swap
from lab.domain.synthetic import Synthetic
from lab.domain.tasks import CRASP, MQAR, AlternatingBlocks, Dyck, Lookup, TypedDyck
from lab.infrastructure.benchmarks.synthetic.sampling import uniform_draws
from lab.infrastructure.benchmarks.synthetic.splits import benchmark_splits
from lab.infrastructure.benchmarks.synthetic.vocabulary import IGNORE

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "legacy_splits.json").read_text())
TASKS = {"mqar": MQAR, "lookup": Lookup, "dyck": Dyck, "blocks": AlternatingBlocks, "dyck2": TypedDyck,
         "crasp": CRASP, "histogram": Histogram, "histogram2": DoubleHistogram, "mode": Mode,
         "most_freq": MostFrequent, "copy": Copy, "reverse": Reverse, "sort": Sort, "count": Count,
         "addition": Addition, "parity": Parity, "boolean_and": BooleanAnd, "random_lm": RandomLM}
RENAMED = {"bracket_types": "types", "formula_depth": "depth", "formula_seed": "program", "histogram_bos": "bos",
           "addition_order": "order", "index_hints": "hints", "carry_sampling": "carries", "and_shift": "shift",
           "and_region": "region"}


def block(task):
    """The task block of a historical task specification (`specs.TaskSpec`)."""
    kind = TASKS[task["task"]]
    names = {field.name for field in fields(kind)}
    return kind(**{RENAMED.get(key, key): value for key, value in task.items() if RENAMED.get(key, key) in names})


def benchmark(record):
    task, study = record["task"], record["study"]
    return Synthetic(task=block(task), length=task["length"], min_length=task["min_length"], train=record["train"],
                     validation=record["validation"], test=record["test"], ood=tuple(record["ood"]),
                     study=None if study is None else Memorization(**study))


def sampled(record):
    """Each split's fingerprint by its historical name, and how many training labels the noise changed."""
    splits, noise = benchmark_splits(benchmark(record), record["data_seed"])
    names = {"test": "in_distribution"}
    return ({names.get(name, name.removeprefix("test/")): split.fingerprint for name, split in splits.items()},
            noise and noise["changed_targets"])


class HistoricalSplitTests(unittest.TestCase):
    def test_every_archived_split(self):
        for record in FIXTURE["runs"]:
            with self.subTest(run=record["runs"][0]):
                fingerprints, changed = sampled(record)
                # The runs did not record their validation probes' fingerprints.
                self.assertEqual({name: value for name, value in fingerprints.items()
                                  if not name.startswith("validation/")}, record["fingerprints"])
                self.assertEqual(changed, record["changed"])

    def test_every_control_of_the_historical_generator(self):
        for record in FIXTURE["grid"]:
            with self.subTest(task=record["task"]["task"], study=record["study"]):
                self.assertEqual(sampled(record), (record["fingerprints"], record["changed"]))

    def test_every_task_is_exercised(self):
        self.assertEqual({record["task"]["task"] for record in FIXTURE["runs"]}, set(TASKS))


class SyntheticSpecTests(unittest.TestCase):
    def test_every_length_a_benchmark_samples_is_checked_when_written(self):
        # Each invalid benchmark, and the nearest valid one.
        cases = ((MQAR(), 16, None, (), MQAR(), 21), (Dyck(), 6, None, (), Dyck(), 9),
                 (Copy(unique=True, symbols=8), 8, None, (16,), Copy(unique=True, symbols=16), 8),
                 (Count(number_limit=12), 8, None, (16,), Count(number_limit=16), 8),
                 (Addition(carry_length=3), 4, 2, (), Addition(carry_length=2), 4))
        for invalid, length, minimum, ood, valid, valid_length in cases:
            with self.subTest(task=valid):
                Synthetic(task=valid, length=valid_length, min_length=minimum, ood=ood)
                with self.assertRaises(ValueError):
                    Synthetic(task=invalid, length=length, min_length=minimum, ood=ood)

    def test_the_random_control_takes_no_noise_and_no_disjoint_pools(self):
        for study in (Memorization(noise=0.1), Memorization(disjoint=True)):
            with self.subTest(study=study), self.assertRaises(ValueError):
                Synthetic(task=RandomLM(), length=8, study=study)
        Synthetic(task=RandomLM(), length=8, study=Memorization())

    def test_held_out_distributions_are_named_as_they_were(self):
        self.assertEqual(list(Synthetic(task=BooleanAnd(), length=8, ood=(16,)).probes),
                         ["length-16", "position_shift", "position-shift-length-16"])
        self.assertEqual(list(Synthetic(task=Addition(), length=4, ood=(8,)).probes),
                         ["length-8", "hard-carry-length-4", "hard-carry-length-8"])
        self.assertEqual(Synthetic(task=Parity(scratchpad="ones", hints=True), length=8, ood=(16,)).context, 67)

    def test_a_selected_held_out_distribution_is_validated_at_the_size_of_validation(self):
        base = Synthetic(task=AlternatingBlocks(), length=16, ood=(32,), train=8, validation=8, test=4)
        selected = swap(base, "select", "length-32")
        self.assertEqual((selected.observed, selected.selection),
                         (("validation", "validation/length-32"), "validation/length-32"))
        splits, _ = benchmark_splits(selected, 0)
        studied, _ = benchmark_splits(swap(base, "study", Memorization()), 0)
        self.assertEqual(len(splits["validation/length-32"].examples), 8)
        self.assertEqual(splits["validation/length-32"].fingerprint, studied["validation/length-32"].fingerprint)
        self.assertEqual(set(splits) - set(benchmark_splits(base, 0)[0]), {"validation/length-32"})
        with self.assertRaisesRegex(ValueError, "select names a held-out distribution"):
            swap(base, "select", "length-64")

    def test_the_final_answer_ranks_first_when_it_is_the_target(self):
        metrics = {"final_answer_accuracy": 0.5, "sequence_accuracy": 0.25, "balanced_accuracy": 0.75, "loss": 2.0}
        final = Synthetic(task=Parity(scratchpad="running"), length=8, metric="final_answer_accuracy")
        self.assertEqual(final.rank(metrics), (0.5, 0.25, 0.75, -2.0))
        self.assertEqual(Synthetic(task=Parity(scratchpad="running"), length=8).rank(metrics), (0.25, 0.75, -2.0))
        with self.assertRaisesRegex(ValueError, "generated answers only"):
            Synthetic(task=Dyck(), length=16, metric="final_answer_accuracy")


class RewriteTests(unittest.TestCase):
    def test_a_query_is_answered_with_the_latest_write_of_its_key(self):
        task = MQAR(symbols=8, pairs=12, queries=4, overwrites=6)
        splits, _ = benchmark_splits(Synthetic(task=task, length=40, train=64, validation=8, test=8), 0)
        answered = rewritten = 0
        for example in splits["train"].examples:
            writes = [tuple(example.tokens[position:position + 2]) for position in range(1, 25, 2)]
            latest = dict(writes)
            self.assertEqual(len(latest), 6)
            current = {}
            for key, value in writes:
                self.assertNotEqual(current.get(key), value)
                current[key] = value
            for position, target in enumerate(example.targets):
                if target != IGNORE:
                    query = example.tokens[position]
                    self.assertEqual(target, latest[query])
                    answered += 1
                    rewritten += sum(key == query for key, _ in writes) > 1
        self.assertEqual(answered, 4 * 64)
        self.assertGreater(rewritten, answered // 2)

    def test_rewrites_leave_a_distinct_key_for_every_query(self):
        MQAR(symbols=4, pairs=8, queries=4, overwrites=4)
        for invalid in ({"pairs": 8, "overwrites": 8}, {"pairs": 8, "queries": 5, "overwrites": 4},
                        {"symbols": 4, "pairs": 8, "overwrites": 3}):
            with self.subTest(task=invalid), self.assertRaises(ValueError):
                MQAR(**invalid)


class SamplingTests(unittest.TestCase):
    def test_uniform_draws_are_randrange_on_the_same_stream(self):
        # A width that is a power of two, one included, rejects half its draws.
        for start, stop in ((0, 1), (7, 8), (272, 528), (-3, 1000), (0, 2 ** 70 + 1)):
            with self.subTest(start=start, stop=stop):
                drawn, expected = random.Random(5), random.Random(5)
                self.assertEqual(uniform_draws(drawn, start, stop, 300),
                                 [expected.randrange(start, stop) for _ in range(300)])
                self.assertEqual(drawn.getstate(), expected.getstate())
        with self.assertRaises(ValueError):
            uniform_draws(random.Random(0), 3, 3, 1)


if __name__ == "__main__":
    unittest.main()
