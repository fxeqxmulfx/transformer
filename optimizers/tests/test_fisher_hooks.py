"""Failed AdaFisher construction must leave the model's existing hooks intact."""

from __future__ import annotations

import unittest
from unittest import mock

from torch import nn

import lean_optim as opt


def _hooks(model: nn.Module) -> list[dict[int, object]]:
    """Snapshot every forward hook, including user-owned registrations."""
    return [dict(layer._forward_hooks) for layer in model.modules()]  # noqa: SLF001 - Check hook ownership after failure.


class FisherHookTests(unittest.TestCase):
    """Keep rejected constructors from retaining factor collectors on a live model."""

    def test_invalid_later_layer_preserves_existing_hooks(self) -> None:
        """A bias in a later linear layer must not leave hooks on earlier valid layers."""
        model = nn.Sequential(nn.Linear(3, 3, bias=False), nn.Linear(3, 2))
        with model[0].register_forward_hook(mock.Mock(return_value=None)):
            before = _hooks(model)
            with self.assertRaisesRegex(NotImplementedError, "bias-free"):
                opt.AdaFisher(model)
            self.assertEqual(_hooks(model), before)

    def test_failed_registration_removes_only_the_new_hooks(self) -> None:
        """Rollback must also handle registration failures after one or more successes."""
        for rejected in (1, 2):
            with self.subTest(layer=rejected):
                model = nn.Sequential(*(nn.Linear(3, 3, bias=False) for _ in range(3)))
                with model[0].register_forward_hook(mock.Mock(return_value=None)):
                    before = _hooks(model)
                    with (
                        mock.patch.object(
                            model[rejected],
                            "register_forward_hook",
                            side_effect=RuntimeError("hook rejected"),
                        ),
                        self.assertRaisesRegex(RuntimeError, "hook rejected"),
                    ):
                        opt.AdaFisher(model)
                    self.assertEqual(_hooks(model), before)
                    optimizer = opt.AdaFisher(model)
                    optimizer.close()
                    self.assertEqual(_hooks(model), before)


if __name__ == "__main__":
    unittest.main()
