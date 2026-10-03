import unittest

from lab.domain.experiment import differences, grid, require_continuation
from lab.domain.model import Norm, Sparsemax, Softmax
from lab.domain.spec import describe, fingerprint, substitute, swap, walk
from lab.domain.training import Budget, CudaGraph
from lab.dsl import GELU, SGD, AMSGradMD, Guarded, LayerNorm, Magma, RMSNorm, Seeds

from examples import gptmini, modular, reference


class SpecTests(unittest.TestCase):
    def setUp(self):
        self.base = modular(gptmini())

    def test_kinds_are_slots_not_blocks(self):
        with self.assertRaisesRegex(TypeError, "RMSNorm, LayerNorm"):
            Norm()

    def test_swap_replaces_one_field_and_rechecks(self):
        changed = swap(self.base, "model.block.attention.weights", Sparsemax())
        self.assertEqual(changed.model.block.attention.weights, Sparsemax())
        self.assertEqual(self.base.model.block.attention.weights, Softmax())
        self.assertEqual(differences(describe(self.base), describe(changed)),
                         ["model.block.attention.weights"])
        with self.assertRaisesRegex(TypeError, "weights must be a Weights block"):
            swap(self.base, "model.block.attention.weights", RMSNorm())
        with self.assertRaisesRegex(KeyError, "no field 'blocks'"):
            swap(self.base, "model.blocks", None)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            swap(self.base, "model.width", 130)

    def test_substitute_replaces_every_occurrence(self):
        changed = substitute(self.base, RMSNorm, LayerNorm())
        norms = [path for path, block in walk(changed) if isinstance(block, LayerNorm)]
        self.assertEqual(norms, ["model.block.norm", "model.final_norm"])
        self.assertEqual(substitute(self.base, RMSNorm(), LayerNorm()), changed)
        with self.assertRaisesRegex(LookupError, "GELU"):
            substitute(self.base, GELU, GELU())

    def test_a_stage_applies_once_at_any_depth(self):
        with self.assertRaisesRegex(ValueError, "MAGMA applies once"):
            Magma(Guarded(Magma(SGD(lr=0.1))))
        with self.assertRaisesRegex(ValueError, "The guard applies once"):
            Guarded(Magma(Guarded(SGD(lr=0.1))))
        self.assertEqual(Guarded(Magma(SGD(lr=0.1))).lr, 0.1)

    def test_magma_damps_directions_which_amsgradmd_does_not_compute(self):
        for spec in (AMSGradMD(lr=1e-3), Guarded(AMSGradMD(lr=0.3, direction_rate=3e-4), sigma=0.25)):
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, "AMSGradMD writes new values"):
                Magma(spec)
        self.assertEqual(Guarded(AMSGradMD(lr=0.3, direction_rate=3e-4), sigma=0.25).lr, 0.3)

    def test_description_identifies_every_field(self):
        description = describe(self.base)
        self.assertEqual(description["model"]["block"]["attention"]["weights"], {"type": "Softmax"})
        self.assertEqual(description["optimizer"]["betas"], [0.9, 0.98])
        variants = [self.base, swap(self.base, "seeds", Seeds(model=1)),
                    swap(self.base, "optimizer.eps", 1e-7), swap(self.base, "execution", CudaGraph())]
        self.assertEqual(len({fingerprint(variant) for variant in variants}), len(variants))
        self.assertEqual(fingerprint(self.base), fingerprint(modular(gptmini())))

    def test_grid_crosses_labeled_alternatives(self):
        variants = grid(self.base, {"model.block.attention.weights": {"soft": Softmax(), "sparse": Sparsemax()},
                                    "seeds.model": {"s0": 0, "s1": 1}})
        self.assertEqual(list(variants), ["soft-s0", "soft-s1", "sparse-s0", "sparse-s1"])
        self.assertEqual(variants["sparse-s1"].seeds.batch_seed, 10_001)
        self.assertEqual(variants["sparse-s1"].model.block.attention.weights, Sparsemax())

    def test_grid_crosses_labeled_bases(self):
        arms = {"soft": self.base, "sparse": substitute(self.base, Softmax, Sparsemax())}
        variants = grid(arms, {"seeds.model": {"s0": 0, "s1": 1}})
        self.assertEqual(list(variants), ["soft-s0", "soft-s1", "sparse-s0", "sparse-s1"])
        self.assertEqual(variants["sparse-s1"], swap(arms["sparse"], "seeds.model", 1))

    def test_continuation_allows_only_a_larger_budget(self):
        stored = describe(self.base)
        require_continuation(stored, swap(self.base, "budget", Budget(updates=300_000, batch=512)))
        with self.assertRaisesRegex(ValueError, "never shorten"):
            require_continuation(stored, swap(self.base, "budget.updates", 1000))
        with self.assertRaisesRegex(ValueError, "changed: budget.batch"):
            require_continuation(stored, swap(self.base, "budget.batch", 1024))
        with self.assertRaisesRegex(ValueError, "changed: model"):
            require_continuation(stored, modular(reference()))

    def test_historical_models_are_valid_compositions(self):
        self.assertEqual(modular(reference()).model.head_width, 32)
        with self.assertRaisesRegex(ValueError, "even head width"):
            gptmini(width=12, heads=4)
        reference(width=12, heads=4)


if __name__ == "__main__":
    unittest.main()
