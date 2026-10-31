"""Retained states preserve a real Schedule-Free trajectory and discard stale futures."""

from pathlib import Path
import tempfile
import unittest

import torch

from lab.application.study import Study, run_study
from lab.dsl import (Budget, Checkpoint, CheckpointHistory, Eager, Evaluate, Experiment,
                     PeriRotate90, PowerDivision, Schedule, ScheduleFreeAdamW, Seeds, describe, swap)
from lab.infrastructure.engine import Engine
from lab.infrastructure.store import RunDirectories, RunDirectory


def recipe():
    return Experiment(model=PeriRotate90(8, 1), benchmark=PowerDivision(7, 2),
                      optimizer=ScheduleFreeAdamW(lr=.003, betas=(.9, .98), weight_decay=1.),
                      schedule=Schedule(warmup=2), budget=Budget(6, 8), seeds=Seeds(31, 11, 11),
                      evaluate=Evaluate(2, 16), checkpoint=CheckpointHistory(2), execution=Eager(device="cpu"))


def train(spec, root):
    return run_study(Study("history", "# checkpoint history\n", {"run": spec}), [],
                     RunDirectories(root), Engine(), lambda *_: None)


class CheckpointHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_retention_does_not_change_training_and_resume_matches(self):
        with tempfile.TemporaryDirectory() as plain, tempfile.TemporaryDirectory() as kept:
            spec = recipe()
            train(swap(spec, "checkpoint", Checkpoint(2)), plain)
            train(swap(spec, "budget.updates", 4), kept)
            train(spec, kept)
            first, second = RunDirectory(Path(plain) / "run"), RunDirectory(Path(kept) / "run")
            a, b = first.checkpoint("cpu"), second.checkpoint("cpu")
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
            torch.testing.assert_close(a["permutation"], b["permutation"], rtol=0, atol=0)
            for key in ("rotary_random_state", "cursor"):
                self.assertEqual(a[key], b[key])
            for index, state in a["optimizer"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, b["optimizer"]["state"][index][name], rtol=0, atol=0)
            snapshots = sorted((second.path / "checkpoints").glob("*.pt"))
            self.assertEqual([torch.load(p, weights_only=True)["step"] for p in snapshots], [0, 2, 4, 6])
            self.assertFalse((first.path / "checkpoints").exists())
            zero = torch.load(snapshots[0], weights_only=True)
            self.assertEqual(zero["optimizer"]["param_groups"][0]["step"], 0)
            self.assertTrue(any(not torch.equal(zero["model"][name], b["model"][name]) for name in b["model"]))

    def test_rewind_removes_only_stale_numbered_snapshots(self):
        with tempfile.TemporaryDirectory() as root:
            run = RunDirectory(root)
            for step in (0, 2, 4):
                run.save({"step": step, "value": torch.tensor([step])}, retain=True)
            note = run.path / "checkpoints" / "notes.pt"
            note.write_text("not a numbered checkpoint")
            run.rewind(2)
            self.assertEqual(sorted(p.name for p in note.parent.iterdir()),
                             ["notes.pt", "step-00000000.pt", "step-00000002.pt"])

    def test_old_checkpoint_description_is_unchanged(self):
        self.assertEqual(describe(Checkpoint(50)), {"type": "Checkpoint", "every": 50})
        self.assertEqual(describe(CheckpointHistory(50)), {"type": "CheckpointHistory", "every": 50})
        with self.assertRaises(ValueError):
            CheckpointHistory(0)


if __name__ == "__main__":
    unittest.main()
