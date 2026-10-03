from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from lab.interfaces.cli import main

PAIR = '''
from lab.dsl import *

model = Transformer(
    width=32, depth=1, context=50,
    block=Block(attention=Attention(heads=2, projections=FusedQKV(), scores=QKNorm(),
                                    weights=Softmax(), exclusive=XSA()),
                ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm()),
    positions=RoPE(), readout=Tied(), final_norm=RMSNorm(), init=Normal(0.02))
base = Experiment(model=model, benchmark=ModularDivision(prime=7, train_fraction=0.5),
                  optimizer=AdamW(lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0),
                  schedule=Schedule(warmup=10), budget=Budget(updates=20, batch=8),
                  seeds=Seeds(), evaluate=Evaluate(every=10), execution=Eager(device="cpu"))
experiments = {"softmax": base, "sparsemax": substitute(base, Softmax, Sparsemax())}
'''


class CliTests(unittest.TestCase):
    def output(self, *arguments):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(main(list(arguments)), 0)
        return stream.getvalue()

    def file(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "pair.py"
        path.write_text(text)
        return str(path)

    def test_check_lists_experiments_and_their_differences(self):
        lines = self.output("check", self.file(PAIR)).splitlines()
        self.assertEqual([line.split()[0] for line in lines], ["softmax", "sparsemax"])
        self.assertEqual(lines[0].split()[-1], "-")
        self.assertEqual(lines[1].split()[-1], "model.block.attention.weights")

    def test_show_prints_the_full_description(self):
        description = json.loads(self.output("show", self.file(PAIR), "sparsemax"))
        self.assertEqual(description["model"]["block"]["attention"]["weights"], {"type": "Sparsemax"})
        with self.assertRaisesRegex(KeyError, "defines no cosine"):
            self.output("show", self.file(PAIR), "cosine")

    def test_files_must_define_labeled_experiments(self):
        with self.assertRaisesRegex(LookupError, "no `experiments`"):
            self.output("check", self.file("x = 1\n"))
        with self.assertRaisesRegex(ValueError, "Label 'Soft max'"):
            self.output("check", self.file(PAIR + "experiments = {'Soft max': base}\n"))

    def test_run_trains_each_experiment_once(self):
        path = self.file(PAIR)
        lines = self.output("run", path, "softmax").splitlines()
        self.assertEqual([line.split()[:3] for line in lines[:3]],
                         [["softmax", "step", "0"], ["softmax", "step", "10"], ["softmax", "step", "20"]])
        self.assertEqual(lines[-1].split()[:4], ["softmax", "finished", "20", "updates"])
        self.assertTrue((Path(path).parent / "runs" / "pair" / "softmax" / "checkpoint.pt").exists())
        self.assertFalse((Path(path).parent / "runs" / "pair" / "sparsemax").exists())
        self.assertEqual(self.output("run", path, "softmax").splitlines(), lines[-1:])

    def test_blocks_lists_every_slot(self):
        listing = self.output("blocks")
        for word in ("Weights:", "Sparsemax()", "CudaGraph(device='cuda', threads=1)", "Composites:"):
            self.assertIn(word, listing)


if __name__ == "__main__":
    unittest.main()
