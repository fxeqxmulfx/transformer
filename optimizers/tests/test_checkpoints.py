"""Checkpoint continuation across dtype and device changes for shared states."""

from __future__ import annotations

import copy
import unittest
from typing import TYPE_CHECKING

import torch

import lean_optim as opt

if TYPE_CHECKING:
    from collections.abc import Callable


def _factories() -> dict[str, Callable[[list[torch.Tensor]], torch.optim.Optimizer]]:
    """Exercise every shared-state layout and mixed-parameter MAGMA factors."""
    return {
        "Dash": lambda p: opt.Dash(p, lr=0.01, block=2, solver=opt.NewtonDB(3)),
        "Guarded": lambda p: opt.Guarded(opt.Adam(p, lr=0.01)),
        "Magma": lambda p: opt.Magma(opt.Adam(p, lr=0.01), updates=20, seed=17),
        "Magma-SGD": lambda p: opt.Magma(opt.SGD(p, lr=0.01), updates=20, seed=17),
        "AMSGradMD-guard": lambda p: opt.AMSGradMD(p, lr=0.01, guard_sigma=0.5),
        "ClippedOptimistic": opt.ClippedOptimistic,
    }


def _parameters(name: str, dtype: torch.dtype, device: str) -> list[torch.Tensor]:
    """Construct matching layouts with the optimistic method's zero start."""
    return [
        torch.nn.Parameter(
            torch.zeros(shape, dtype=dtype, device=device)
            if name == "ClippedOptimistic"
            else torch.arange(1, 1 + torch.Size(shape).numel(), dtype=dtype, device=device)
            .reshape(shape)
            .div_(8)
        )
        for shape in ((2, 3), (2,))
    ]


def _step(optimizer: torch.optim.Optimizer, sign: float) -> None:
    """Supply identical, exactly representable gradients without a model dependency."""
    for parameter in optimizer.param_groups[0]["params"]:
        parameter.grad = torch.arange(
            1, parameter.numel() + 1, dtype=parameter.dtype, device=parameter.device
        ).reshape_as(parameter)
        parameter.grad.mul_(sign / 16)
    optimizer.step()


class CheckpointTests(unittest.TestCase):
    """Check values, target placement, counters and continuation after loading."""

    def _continuation(
        self,
        name: str,
        factory: Callable[[list[torch.Tensor]], torch.optim.Optimizer],
        source_placement: tuple[torch.dtype, str],
        target_placement: tuple[torch.dtype, str],
    ) -> None:
        source = _parameters(name, *source_placement)
        reference = _parameters(name, *target_placement)
        original, control = factory(source), factory(reference)
        for sign in (1.0, -0.5, 0.0):
            _step(original, sign)
            _step(control, sign)
        checkpoint = copy.deepcopy(original.state_dict())
        checkpoint_before = copy.deepcopy(checkpoint)
        parameters = _parameters(name, *target_placement)
        restored = factory(parameters)
        # Warm the destination too: loading must invalidate its old scratch.
        _step(restored, 0.25)
        layouts = {
            key: {name: (tensor.device, tensor.dtype) for name, tensor in state.items()}
            for key, state in restored.state.items()
            if isinstance(key, str)
        }
        with torch.no_grad():
            for parameter, saved in zip(parameters, source, strict=True):
                parameter.copy_(saved)
        restored.load_state_dict(checkpoint)
        self.assertFalse(restored.workspace.buffers)
        for key, layout in layouts.items():
            for field, (device, dtype) in layout.items():
                loaded = restored.state[key][field]
                self.assertEqual(loaded.device, device)
                self.assertEqual(loaded.dtype, dtype)
                torch.testing.assert_close(
                    loaded, checkpoint_before["state"][key][field].to(device=device, dtype=dtype)
                )
                # Loading must not rewrite the caller's shared-state dictionary.
                self.assertEqual(
                    checkpoint["state"][key][field].device,
                    checkpoint_before["state"][key][field].device,
                )
                self.assertEqual(
                    checkpoint["state"][key][field].dtype,
                    checkpoint_before["state"][key][field].dtype,
                )
        for sign in (-1.0, 0.25, 0.5):
            _step(restored, sign)
            _step(control, sign)
            for parameter, expected in zip(parameters, reference, strict=True):
                torch.testing.assert_close(
                    parameter.detach().cpu().double(),
                    expected.detach().cpu().double(),
                    rtol=2e-5,
                    atol=2e-6,
                )
        self._check_counters(restored, control)

    def _check_counters(
        self, restored: torch.optim.Optimizer, control: torch.optim.Optimizer
    ) -> None:
        for key in ("guard", "magma"):
            for field in ("updates", "accepted"):
                value = restored.state.get(key, {}).get(field)
                if value is not None:
                    self.assertEqual(int(value), int(control.state[key][field]))

    def test_shared_histories_follow_target_dtype_and_keep_integer_counters(self) -> None:
        """Downcast and upcast histories while retaining int64 counters."""
        for name, factory in _factories().items():
            for source, target in ((torch.float64, torch.float32), (torch.float32, torch.float64)):
                with self.subTest(rule=name, source=source, target=target):
                    self._continuation(name, factory, (source, "cpu"), (target, "cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_shared_histories_follow_cpu_cuda_checkpoint_targets(self) -> None:
        """Resume both CPU-to-CUDA and CUDA-to-CPU without stranded shared tensors."""
        for name, factory in _factories().items():
            for source, target in (("cpu", "cuda"), ("cuda", "cpu")):
                with self.subTest(rule=name, source=source, target=target):
                    self._continuation(
                        name, factory, (torch.float64, source), (torch.float64, target)
                    )

    def test_magma_restores_each_matrix_dtype_in_a_mixed_group(self) -> None:
        """Scalar scales and auxiliary moments follow their own selected matrix."""
        source = [torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float64)) for _ in range(2)]
        original = opt.Magma(opt.SGD(source), updates=10, seed=17)
        _step(original, 1.0)
        target = [
            torch.nn.Parameter(parameter.detach().to(dtype).clone())
            for parameter, dtype in zip(source, (torch.float32, torch.float64), strict=True)
        ]
        restored = opt.Magma(opt.SGD(target), updates=10, seed=17)
        restored.load_state_dict(copy.deepcopy(original.state_dict()))
        state = restored.state["magma"]
        for index, parameter in enumerate(target):
            for field in ("scale", "scale_sum", "cosine_sum", "moment"):
                self.assertEqual(state[f"{index}.{field}"].dtype, parameter.dtype)
        self.assertEqual(state["updates"].dtype, torch.int64)
        _step(restored, -0.5)

    def test_dash_checkpoint_uses_the_promoted_target_dtype_in_both_orders(self) -> None:
        """Restored batched histories use the whole group's promoted dtype."""
        source = _parameters("Dash", torch.float32, "cpu")
        original = opt.Dash(source, block=2, solver=opt.NewtonDB(3))
        _step(original, 1.0)
        for dtypes in ((torch.float32, torch.float64), (torch.float64, torch.float32)):
            with self.subTest(dtypes=dtypes):
                target = [
                    torch.nn.Parameter(parameter.detach().to(dtype).clone())
                    for parameter, dtype in zip(source, dtypes, strict=True)
                ]
                restored = opt.Dash(target, block=2, solver=opt.NewtonDB(3))
                restored.load_state_dict(copy.deepcopy(original.state_dict()))
                self.assertTrue(
                    all(value.dtype == torch.float64 for value in restored.state["dash"].values())
                )
                _step(restored, -0.5)


if __name__ == "__main__":
    unittest.main()
